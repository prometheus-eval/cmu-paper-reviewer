"""POST /api/review/{key}/debate and related debate-with-AI endpoints."""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import get_session, async_session
from backend.models import DebateMessage, DebateSession, Submission, SubmissionStatus
from backend.schemas import (
    DebateStartRequest,
    DebateStartResponse,
    DebateFeedbackRequest,
    DebateMessageRequest,
    DebateSessionResponse,
    DebateMessageResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["debates"])


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _build_debate_system_prompt(key: str, item_number: int) -> str:
    """Build a system prompt for the debate LLM based on the review item and paper."""
    from backend.services.storage_service import get_review_markdown, preprint_md_path
    from backend.services.pdf_service import _parse_review

    # ── Load and parse the review ────────────────────────────────────────────
    review_md = get_review_markdown(key)
    if not review_md:
        raise HTTPException(status_code=404, detail="Review markdown not found.")

    parsed = _parse_review(review_md)

    # Find the specific item by number
    target_item = None
    for item in parsed.items:
        if item.number == item_number:
            target_item = item
            break

    if target_item is None:
        raise HTTPException(
            status_code=404,
            detail=f"Review item #{item_number} not found.",
        )

    # ── Format the review item for the prompt ────────────────────────────────
    evidence_text = ""
    for i, ev in enumerate(target_item.evidence, 1):
        evidence_text += f"  Quote {i}: {ev.quote}\n"
        evidence_text += f"  Comment {i}: {ev.comment}\n\n"

    item_block = (
        f"Title: {target_item.title}\n"
        f"Main point of criticism: {target_item.main_criticism}\n"
        f"Evaluation criteria: {target_item.eval_criteria}\n"
        f"Evidence:\n{evidence_text}"
    )

    # ── Load paper markdown (truncated) ──────────────────────────────────────
    paper_md_path = preprint_md_path(key)
    if paper_md_path.exists():
        paper_md = paper_md_path.read_text(encoding="utf-8")
    else:
        paper_md = "(Paper content unavailable.)"

    if len(paper_md) > 30000:
        paper_md = paper_md[:30000] + "\n\n[... truncated ...]"

    # ── Assemble system prompt ───────────────────────────────────────────────
    system_prompt = (
        "You are a rigorous academic reviewer defending a specific criticism you "
        "raised about a research paper. A human author is now debating you.\n\n"
        "## Your Review Item\n"
        f"{item_block}\n"
        "## Paper Content\n"
        f"{paper_md}\n\n"
        "## Instructions\n"
        "- Defend your criticism with evidence from the paper.\n"
        "- Be non-sycophantic. Do NOT concede just to be agreeable.\n"
        "- Only concede if the author successfully refutes ALL evidence "
        "supporting your criticism.\n"
        "- Stay focused on the specific review item above.\n\n"
        "## Rules\n"
        '- If the author goes off-topic, respond with exactly: DERAIL\n'
        '- On your final turn (when instructed), you MUST end your response '
        'with exactly one of:\n'
        '  "I was convinced (I am wrong)"\n'
        '  "I was not convinced (This is a problem)"\n'
    )

    return system_prompt


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("/review/{key}/debate", response_model=DebateStartResponse)
async def start_debate(
    key: str,
    body: DebateStartRequest,
    session: AsyncSession = Depends(get_session),
):
    """Start a new debate session for a specific review item."""
    # Validate submission exists and is completed
    result = await session.execute(select(Submission).where(Submission.key == key))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found.")
    if submission.status != SubmissionStatus.completed:
        raise HTTPException(status_code=400, detail="Review is not yet completed.")

    # Create the debate session
    debate = DebateSession(
        key=key,
        item_number=body.item_number,
        annotator_id=body.annotator_id,
        model_used=submission.review_model_used or "",
        status="active",
        turn_count=0,
    )
    session.add(debate)
    await session.flush()  # populate debate.id

    # Build system prompt and persist as the first message
    system_prompt = _build_debate_system_prompt(key, body.item_number)
    system_msg = DebateMessage(
        session_id=debate.id,
        role="system",
        content=system_prompt,
        turn_number=0,
    )
    session.add(system_msg)
    await session.commit()

    return DebateStartResponse(session_id=debate.id)


@router.post("/review/{key}/debate/{session_id}/message")
async def post_debate_message(
    key: str,
    session_id: int,
    body: DebateMessageRequest,
    session: AsyncSession = Depends(get_session),
):
    """Send a user message and stream the AI response via SSE."""
    # Validate session
    result = await session.execute(
        select(DebateSession).where(
            DebateSession.id == session_id,
            DebateSession.key == key,
        )
    )
    debate = result.scalar_one_or_none()
    if not debate:
        raise HTTPException(status_code=404, detail="Debate session not found.")
    if debate.status != "active":
        raise HTTPException(status_code=400, detail="Debate session is no longer active.")
    if debate.turn_count >= 20:
        raise HTTPException(status_code=400, detail="Maximum number of turns reached.")

    # Increment turn count and save user message
    debate.turn_count += 1
    current_turn = debate.turn_count

    user_msg = DebateMessage(
        session_id=session_id,
        role="user",
        content=body.content,
        turn_number=current_turn,
    )
    session.add(user_msg)
    await session.commit()

    # Load all messages for the LLM context
    msg_result = await session.execute(
        select(DebateMessage)
        .where(DebateMessage.session_id == session_id)
        .order_by(DebateMessage.turn_number, DebateMessage.id)
    )
    all_messages = msg_result.scalars().all()

    messages = [{"role": m.role, "content": m.content} for m in all_messages]

    # If we're at turn 20, inject a final verdict instruction
    if current_turn >= 20:
        messages.append({
            "role": "system",
            "content": (
                "This is the final turn of the debate. You MUST end your response "
                'with exactly one of:\n'
                '"I was convinced (I am wrong)"\n'
                '"I was not convinced (This is a problem)"'
            ),
        })

    model_used = debate.model_used
    final_turn = current_turn >= 20

    async def event_stream():
        import litellm

        full_content = ""
        try:
            response = litellm.completion(
                model=model_used,
                api_key=settings.litellm_api_key,
                api_base=settings.litellm_base_url,
                messages=messages,
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta
                token = getattr(delta, "content", None) or ""
                if token:
                    full_content += token
                    yield f"data: {json.dumps({'content': token})}\n\n"

        except Exception as exc:
            logger.exception("LLM streaming error in debate session %s", session_id)
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        # ── Post-stream DB operations using a fresh session ──────────────
        try:
            async with async_session() as db:
                # Reload debate to avoid stale state
                res = await db.execute(
                    select(DebateSession).where(DebateSession.id == session_id)
                )
                db_debate = res.scalar_one_or_none()

                # Save assistant message
                assistant_msg = DebateMessage(
                    session_id=session_id,
                    role="assistant",
                    content=full_content,
                    turn_number=current_turn,
                )
                db.add(assistant_msg)

                # Detect DERAIL
                if "DERAIL" in full_content.strip():
                    if db_debate:
                        db_debate.status = "derailed"

                # Detect final verdict at turn 20
                if final_turn and db_debate:
                    if "I was convinced (I am wrong)" in full_content:
                        db_debate.status = "concluded_convinced"
                    elif "I was not convinced (This is a problem)" in full_content:
                        db_debate.status = "concluded_not_convinced"

                await db.commit()
        except Exception:
            logger.exception(
                "Failed to persist assistant message for debate session %s",
                session_id,
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/review/{key}/debate/{session_id}/feedback")
async def submit_debate_feedback(
    key: str,
    session_id: int,
    body: DebateFeedbackRequest,
    session: AsyncSession = Depends(get_session),
):
    """Record whether the user agrees with the debate outcome."""
    result = await session.execute(
        select(DebateSession).where(
            DebateSession.id == session_id,
            DebateSession.key == key,
        )
    )
    debate = result.scalar_one_or_none()
    if not debate:
        raise HTTPException(status_code=404, detail="Debate session not found.")

    debate.user_agrees_with_result = body.user_agrees
    await session.commit()

    return {"status": "ok"}


@router.get("/review/{key}/debate/{session_id}", response_model=DebateSessionResponse)
async def get_debate_session(
    key: str,
    session_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Return full debate history (excluding system messages)."""
    result = await session.execute(
        select(DebateSession).where(
            DebateSession.id == session_id,
            DebateSession.key == key,
        )
    )
    debate = result.scalar_one_or_none()
    if not debate:
        raise HTTPException(status_code=404, detail="Debate session not found.")

    msg_result = await session.execute(
        select(DebateMessage)
        .where(
            DebateMessage.session_id == session_id,
            DebateMessage.role != "system",
        )
        .order_by(DebateMessage.turn_number, DebateMessage.id)
    )
    db_messages = msg_result.scalars().all()

    return DebateSessionResponse(
        id=debate.id,
        key=debate.key,
        item_number=debate.item_number,
        annotator_id=debate.annotator_id,
        model_used=debate.model_used,
        status=debate.status,
        user_agrees_with_result=debate.user_agrees_with_result,
        turn_count=debate.turn_count,
        messages=[
            DebateMessageResponse(
                id=m.id,
                role=m.role,
                content=m.content,
                turn_number=m.turn_number,
            )
            for m in db_messages
        ],
        created_at=debate.created_at,
        updated_at=debate.updated_at,
    )


@router.get("/review/{key}/debates")
async def list_debate_sessions(
    key: str,
    annotator_id: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    """List all debate sessions for a given review, optionally filtered by annotator."""
    query = select(DebateSession).where(DebateSession.key == key)
    if annotator_id:
        query = query.where(DebateSession.annotator_id == annotator_id)
    query = query.order_by(DebateSession.created_at.desc())

    result = await session.execute(query)
    debates = result.scalars().all()

    return [
        {
            "id": d.id,
            "key": d.key,
            "item_number": d.item_number,
            "annotator_id": d.annotator_id,
            "model_used": d.model_used,
            "status": d.status,
            "user_agrees_with_result": d.user_agrees_with_result,
            "turn_count": d.turn_count,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
        }
        for d in debates
    ]


@router.get("/debates/export")
async def export_all_debates(
    x_admin_key: str = Header(None),
    session: AsyncSession = Depends(get_session),
):
    """Export all debate sessions with their messages as JSON (admin only)."""
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API key not configured on server.")
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key.")

    # Fetch all sessions
    result = await session.execute(
        select(DebateSession).order_by(DebateSession.key, DebateSession.item_number)
    )
    sessions = result.scalars().all()

    data = []
    for s in sessions:
        # Fetch messages for this session (exclude system prompts for brevity)
        msg_result = await session.execute(
            select(DebateMessage)
            .where(DebateMessage.session_id == s.id, DebateMessage.role != "system")
            .order_by(DebateMessage.turn_number, DebateMessage.id)
        )
        messages = msg_result.scalars().all()

        data.append({
            "session_id": s.id,
            "key": s.key,
            "item_number": s.item_number,
            "annotator_id": s.annotator_id,
            "model_used": s.model_used,
            "status": s.status,
            "user_agrees_with_result": s.user_agrees_with_result,
            "turn_count": s.turn_count,
            "created_at": s.created_at.isoformat() if s.created_at else None,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "turn_number": m.turn_number,
                }
                for m in messages
            ],
        })

    return JSONResponse(data)
