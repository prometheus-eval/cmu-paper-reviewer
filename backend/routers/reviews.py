"""GET /api/review/{key} and related endpoints."""

import json

from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import settings
from backend.database import get_session
from backend.models import Annotation, Submission, SubmissionStatus
from backend.schemas import (
    AnnotationRequest,
    AnnotationResponse,
    ReviewResponse,
    VerificationCodeFile,
    VerificationCodeResponse,
)
from backend.services.pdf_service import generate_review_pdf
from backend.services.storage_service import (
    annotations_path,
    get_review_markdown,
    images_dir,
    images_list_path,
    list_verification_code_files,
    preprint_md_path,
    review_pdf_path,
    verification_code_dir,
)

router = APIRouter(prefix="/api", tags=["reviews"])


@router.get("/review/{key}", response_model=ReviewResponse)
async def get_review(
    key: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Submission).where(Submission.key == key))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found.")

    if submission.status != SubmissionStatus.completed:
        return ReviewResponse(
            key=key,
            status=submission.status,
            review_markdown=None,
        )

    md = get_review_markdown(key)
    return ReviewResponse(key=key, status=submission.status, review_markdown=md)


@router.get("/review/{key}/pdf")
async def get_review_pdf(
    key: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Submission).where(Submission.key == key))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found.")

    if submission.status != SubmissionStatus.completed:
        raise HTTPException(status_code=202, detail="Review is not yet complete.")

    pdf_path = review_pdf_path(key)
    if not pdf_path.exists():
        # Generate on the fly if MD exists but PDF hasn't been created yet
        generated = generate_review_pdf(key)
        if not generated:
            raise HTTPException(status_code=404, detail="Review PDF not available.")

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=f"review_{key}.pdf",
    )


@router.get("/review/{key}/paper")
async def get_paper_markdown(
    key: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Submission).where(Submission.key == key))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found.")

    md_path = preprint_md_path(key)
    if not md_path.exists():
        raise HTTPException(status_code=404, detail="OCR'd paper not found.")

    content = md_path.read_text(encoding="utf-8")
    return {"key": key, "paper_markdown": content}


@router.get("/review/{key}/paper/images")
async def get_paper_images_list(
    key: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Submission).where(Submission.key == key))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Submission not found.")

    list_path = images_list_path(key)
    if not list_path.exists():
        return {"key": key, "images": []}

    import json
    data = json.loads(list_path.read_text(encoding="utf-8"))
    return {"key": key, "images": data}


@router.get("/review/{key}/paper/images/{filename}")
async def get_paper_image(
    key: str,
    filename: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Submission).where(Submission.key == key))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Submission not found.")

    img_path = images_dir(key) / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")

    media_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    return FileResponse(path=str(img_path), media_type=media_type, filename=filename)


# ─── Verification Code ──────────────────────────────────────────────────────

@router.get("/review/{key}/verification-code", response_model=VerificationCodeResponse)
async def get_verification_code_list(
    key: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Submission).where(Submission.key == key))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found.")

    files = list_verification_code_files(key)
    return VerificationCodeResponse(
        files=[VerificationCodeFile(name=f["name"], size=f["size"]) for f in files]
    )


@router.get("/review/{key}/verification-code/{file_path:path}")
async def get_verification_code_file(
    key: str,
    file_path: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Submission).where(Submission.key == key))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Submission not found.")

    vdir = verification_code_dir(key)
    target = vdir / file_path
    # Prevent path traversal
    try:
        target.resolve().relative_to(vdir.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path.")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    # Return content as text for display
    try:
        content = target.read_text(encoding="utf-8")
        return JSONResponse({"name": file_path, "content": content})
    except UnicodeDecodeError:
        return FileResponse(path=str(target), filename=target.name)


# ─── Annotations ─────────────────────────────────────────────────────────────

@router.post("/review/{key}/annotations", response_model=AnnotationResponse)
async def submit_annotation(
    key: str,
    body: AnnotationRequest,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Submission).where(Submission.key == key))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found.")

    # Validate values
    valid_correctness = {"correct", "incorrect"}
    valid_significance = {"significant", "marginally_significant", "not_significant"}
    valid_evidence = {"sufficient", "insufficient"}

    if body.correctness and body.correctness not in valid_correctness:
        raise HTTPException(status_code=400, detail=f"correctness must be one of: {valid_correctness}")
    if body.significance and body.significance not in valid_significance:
        raise HTTPException(status_code=400, detail=f"significance must be one of: {valid_significance}")
    if body.evidence_quality and body.evidence_quality not in valid_evidence:
        raise HTTPException(status_code=400, detail=f"evidence_quality must be one of: {valid_evidence}")

    annotator = body.annotator_id or "anonymous"

    # Upsert: check if annotation for this key+item+annotator already exists
    existing = await session.execute(
        select(Annotation).where(
            Annotation.key == key,
            Annotation.item_number == body.item_number,
            Annotation.annotator_id == annotator,
        )
    )
    annotation = existing.scalar_one_or_none()

    if annotation:
        if body.correctness is not None:
            annotation.correctness = body.correctness
        if body.significance is not None:
            annotation.significance = body.significance
        if body.evidence_quality is not None:
            annotation.evidence_quality = body.evidence_quality
    else:
        annotation = Annotation(
            key=key,
            item_number=body.item_number,
            annotator_id=annotator,
            correctness=body.correctness,
            significance=body.significance,
            evidence_quality=body.evidence_quality,
        )
        session.add(annotation)

    await session.commit()

    # Also persist to JSON file for easy download
    _save_annotations_json(key, session)

    return AnnotationResponse(
        key=key,
        item_number=body.item_number,
        annotator_id=annotator,
        correctness=annotation.correctness,
        significance=annotation.significance,
        evidence_quality=annotation.evidence_quality,
    )


@router.get("/review/{key}/annotations")
async def get_annotations(
    key: str,
    annotator_id: str | None = None,
    session: AsyncSession = Depends(get_session),
):
    query = select(Annotation).where(Annotation.key == key)
    if annotator_id:
        query = query.where(Annotation.annotator_id == annotator_id)
    query = query.order_by(Annotation.item_number)

    result = await session.execute(query)
    annotations = result.scalars().all()
    return [
        AnnotationResponse(
            key=a.key,
            item_number=a.item_number,
            annotator_id=a.annotator_id,
            correctness=a.correctness,
            significance=a.significance,
            evidence_quality=a.evidence_quality,
        )
        for a in annotations
    ]


@router.get("/annotations/export")
async def export_all_annotations(
    x_admin_key: str = Header(None),
    session: AsyncSession = Depends(get_session),
):
    """Export all annotations across all submissions as JSON (admin only)."""
    if not settings.admin_api_key:
        raise HTTPException(status_code=503, detail="Admin API key not configured on server.")
    if x_admin_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing admin API key.")

    result = await session.execute(
        select(Annotation).order_by(Annotation.key, Annotation.item_number)
    )
    annotations = result.scalars().all()
    data = [
        {
            "key": a.key,
            "item_number": a.item_number,
            "annotator_id": a.annotator_id,
            "correctness": a.correctness,
            "significance": a.significance,
            "evidence_quality": a.evidence_quality,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in annotations
    ]
    return JSONResponse(data)


def _save_annotations_json(key: str, session):
    """Persist annotations to a JSON file alongside the review."""
    import asyncio

    async def _do():
        result = await session.execute(
            select(Annotation).where(Annotation.key == key).order_by(Annotation.item_number)
        )
        annotations = result.scalars().all()
        data = [
            {
                "key": a.key,
                "item_number": a.item_number,
                "annotator_id": a.annotator_id,
                "correctness": a.correctness,
                "significance": a.significance,
                "evidence_quality": a.evidence_quality,
            }
            for a in annotations
        ]
        path = annotations_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    try:
        asyncio.ensure_future(_do())
    except RuntimeError:
        pass  # No event loop; skip file persistence
