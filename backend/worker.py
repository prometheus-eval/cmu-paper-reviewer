"""Background worker that polls SQLite for pending submissions and processes them.

Run as: python -m backend.worker
"""

import asyncio
import json
import logging
import shutil
import time
import traceback
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, case, delete, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import settings
from backend.models import Annotation, Base, Submission, SubmissionMode, SubmissionStatus
from backend.services.ocr_service import OCRService
from backend.services.pdf_service import generate_review_pdf
from backend.services.review_service import ReviewService
from backend.services.storage_service import review_dir, upload_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Use synchronous engine for the worker (OpenHands agent is synchronous)
sync_url = settings.database_url.replace("sqlite+aiosqlite", "sqlite")
engine = create_engine(sync_url, echo=False)
SessionLocal = sessionmaker(engine, class_=Session)


def get_next_pending() -> Submission | None:
    """Get the next pending submission, prioritizing BYOK over queue."""
    with SessionLocal() as session:
        result = session.execute(
            select(Submission)
            .where(Submission.status == SubmissionStatus.pending)
            .order_by(
                # BYOK first (0), then queue (1)
                case(
                    (Submission.mode == SubmissionMode.byok, 0),
                    else_=1,
                ),
                Submission.created_at,
            )
            .limit(1)
        )
        sub = result.scalar_one_or_none()
        if sub:
            session.expunge(sub)
        return sub


def update_status(key: str, status: SubmissionStatus, error: str | None = None):
    with SessionLocal() as session:
        result = session.execute(select(Submission).where(Submission.key == key))
        sub = result.scalar_one_or_none()
        if sub:
            sub.status = status
            if error:
                sub.error_message = error
            session.commit()


def clear_user_keys(key: str):
    """Clear stored user API keys after processing."""
    with SessionLocal() as session:
        result = session.execute(select(Submission).where(Submission.key == key))
        sub = result.scalar_one_or_none()
        if sub:
            sub.user_mistral_api_key = None
            sub.user_litellm_api_key = None
            sub.user_litellm_base_url = None
            sub.user_tavily_api_key = None
            session.commit()


def store_model_used(key: str, model_name: str):
    """Store which model was used for the review."""
    with SessionLocal() as session:
        result = session.execute(select(Submission).where(Submission.key == key))
        sub = result.scalar_one_or_none()
        if sub:
            sub.review_model_used = model_name
            session.commit()


BUDGET_ERROR_KEYWORDS = [
    "budget has been exceeded",
    "budget exceeded",
    "exceeded your current budget",
    "BudgetExceededError",
    "over budget",
    "out of budget",
    "insufficient budget",
    "rate_limit_error",
    "insufficient_quota",
    "exceeded your current quota",
]


def _is_budget_error(error_text: str) -> bool:
    """Check if an error message indicates a budget/quota issue."""
    lower = error_text.lower()
    return any(kw.lower() in lower for kw in BUDGET_ERROR_KEYWORDS)


def _validate_review(key: str) -> bool:
    """Check that the generated review contains at least one review item."""
    import re
    from backend.services.storage_service import review_md_path
    md_path = review_md_path(key)
    if not md_path.exists():
        return False
    content = md_path.read_text(encoding="utf-8")
    # Must contain at least one "## Item N:" header
    return bool(re.search(r"^##\s*Item\s+\d+\s*:", content, re.MULTILINE | re.IGNORECASE))


def process_submission(submission: Submission):
    key = submission.key
    pdf_file = upload_path(key, submission.filename)
    is_byok = submission.mode == SubmissionMode.byok

    # Parse review settings
    review_settings = None
    if submission.review_settings:
        try:
            review_settings = json.loads(submission.review_settings)
        except (json.JSONDecodeError, TypeError):
            logger.warning("[%s] Invalid review_settings JSON, using defaults.", key)

    try:
        # Step 0: Send "review started" email
        if submission.email:
            try:
                from backend.services.email_service import send_review_started_email
                asyncio.run(send_review_started_email(submission.email, key, submission.filename))
                logger.info("[%s] 'Review started' email sent to %s.", key, submission.email)
            except Exception:
                logger.exception("[%s] 'Review started' email failed (non-critical).", key)

        # Step 1: OCR
        logger.info("[%s] Starting OCR... (mode=%s)", key, submission.mode.value)
        update_status(key, SubmissionStatus.ocr)
        ocr = OCRService(api_key=submission.user_mistral_api_key if is_byok else None)
        ocr.process_pdf(str(pdf_file), key)
        logger.info("[%s] OCR complete.", key)

        # Step 1.5: Extract paper date (for reference filtering)
        if review_settings is None:
            review_settings = {}
        if not review_settings.get("paper_date"):
            try:
                from backend.services.paper_date_service import get_paper_date
                from backend.services.storage_service import preprint_md_path
                md_path = preprint_md_path(key)
                if md_path.exists():
                    md_text = md_path.read_text(encoding="utf-8")
                    paper_date = get_paper_date(md_text, filename=submission.filename)
                    if paper_date:
                        review_settings["paper_date"] = paper_date
                        logger.info("[%s] Extracted paper date: %s", key, paper_date)
                    else:
                        logger.info("[%s] Could not determine paper date.", key)
            except Exception:
                logger.exception("[%s] Paper date extraction failed (non-critical).", key)

        # Step 2: Review (with validation and retry)
        logger.info("[%s] Starting review...", key)
        update_status(key, SubmissionStatus.reviewing)

        max_attempts = 2
        model_used = None
        for attempt in range(1, max_attempts + 1):
            reviewer = ReviewService(
                litellm_api_key=submission.user_litellm_api_key if is_byok else None,
                litellm_base_url=submission.user_litellm_base_url if is_byok else None,
                tavily_api_key=submission.user_tavily_api_key if is_byok else None,
                review_settings=review_settings,
            )
            review_path, model_used = reviewer.run_review(key)
            store_model_used(key, model_used)
            logger.info("[%s] Review attempt %d complete. Model: %s", key, attempt, model_used)

            # Validate: review must contain at least one "## Item" header
            if _validate_review(key):
                break
            else:
                logger.warning("[%s] Review validation failed (attempt %d/%d) — no review items found.",
                               key, attempt, max_attempts)
                if attempt < max_attempts:
                    # Delete the bad review so the retry starts fresh
                    from backend.services.storage_service import review_md_path as _rmp
                    bad = _rmp(key)
                    if bad.exists():
                        bad.unlink()
                    logger.info("[%s] Retrying with a different model...", key)
                else:
                    logger.error("[%s] All %d review attempts produced invalid output.", key, max_attempts)

        # Step 3: Generate PDF
        logger.info("[%s] Generating PDF...", key)
        generate_review_pdf(key, model_name=model_used or "")
        logger.info("[%s] PDF generated.", key)

        # Step 4: Mark as completed
        update_status(key, SubmissionStatus.completed)
        logger.info("[%s] Submission complete!", key)

        # Step 5: Send email notification (only if email was provided)
        if submission.email:
            try:
                from backend.services.email_service import send_review_ready_email
                result = asyncio.run(send_review_ready_email(submission.email, key))
                if result:
                    logger.info("[%s] Email sent to %s.", key, submission.email)
                else:
                    logger.warning("[%s] Email send returned False (SMTP not configured or send failed).", key)
            except Exception:
                logger.exception("[%s] Email notification failed.", key)
    finally:
        # Always clear stored user API keys after processing
        if is_byok:
            clear_user_keys(key)
            logger.info("[%s] User API keys cleared.", key)


CLEANUP_MAX_AGE = timedelta(hours=24)


def _cleanup_annotated(sub):
    """For annotated submissions, delete uploaded PDF and non-essential files.

    Keeps: OCR'd markdown, images, images_list.json, review markdown/PDF,
           annotations JSON.
    Deletes: uploaded PDF, code, supplementary, verification code, trajectory.
    """
    # Delete uploaded PDF
    upload_f = upload_path(sub.key, sub.filename)
    if upload_f.exists():
        upload_f.unlink(missing_ok=True)
        logger.info("[%s] Deleted uploaded PDF to save space.", sub.key)

    # Delete code directory
    code_d = review_dir(sub.key) / "preprint" / "code"
    if code_d.exists():
        shutil.rmtree(code_d, ignore_errors=True)

    # Delete supplementary directory
    supp_d = review_dir(sub.key) / "preprint" / "supplementary"
    if supp_d.exists():
        shutil.rmtree(supp_d, ignore_errors=True)

    # Delete verification code and trajectory directories
    review_out = review_dir(sub.key) / "review"
    if review_out.exists():
        for child in review_out.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)


def cleanup_old_submissions():
    """Delete submissions and their files older than CLEANUP_MAX_AGE.

    - No annotations: delete everything (files + DB record).
    - Has annotations: delete uploaded PDF and non-essential files, keep
      OCR artifacts, review, and annotations.
    """
    cutoff = datetime.now(timezone.utc) - CLEANUP_MAX_AGE
    with SessionLocal() as session:
        annotated_keys = set(
            row[0] for row in session.execute(
                select(Annotation.key).distinct()
            ).all()
        )

        old = session.execute(
            select(Submission).where(Submission.created_at < cutoff)
        ).scalars().all()

        if not old:
            return

        to_delete_keys = []
        for sub in old:
            if sub.key in annotated_keys:
                _cleanup_annotated(sub)
                continue
            # No annotations — delete everything
            review_d = review_dir(sub.key)
            if review_d.exists():
                shutil.rmtree(review_d, ignore_errors=True)
            upload_f = upload_path(sub.key, sub.filename)
            if upload_f.exists():
                upload_f.unlink(missing_ok=True)
            to_delete_keys.append(sub.key)
            logger.info("[%s] Cleaned up old submission (age > %s).", sub.key, CLEANUP_MAX_AGE)

        if to_delete_keys:
            session.execute(
                delete(Submission).where(Submission.key.in_(to_delete_keys))
            )
            session.commit()


def recover_stuck_submissions():
    """Reset submissions stuck in 'ocr' or 'reviewing' back to 'pending'."""
    with SessionLocal() as session:
        stuck = session.execute(
            select(Submission).where(
                Submission.status.in_([SubmissionStatus.ocr, SubmissionStatus.reviewing])
            )
        ).scalars().all()
        for sub in stuck:
            sub.status = SubmissionStatus.pending
            sub.error_message = None
            logger.info("[%s] Recovered stuck submission (was %s).", sub.key, sub.status)
        if stuck:
            session.commit()


def _migrate_add_columns():
    """Add new columns to existing tables if they don't exist (lightweight migration)."""
    import sqlite3
    db_path = sync_url.replace("sqlite:///", "")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Get existing columns
    cursor.execute("PRAGMA table_info(submissions)")
    existing = {row[1] for row in cursor.fetchall()}
    new_cols = {
        "client_ip": "VARCHAR(45)",
        "review_settings": "TEXT",
        "review_model_used": "TEXT",
    }
    for col, col_type in new_cols.items():
        if col not in existing:
            cursor.execute(f"ALTER TABLE submissions ADD COLUMN {col} {col_type}")
            logger.info("Added column submissions.%s", col)

    # Annotations table migration
    cursor.execute("PRAGMA table_info(annotations)")
    ann_existing = {row[1] for row in cursor.fetchall()}
    if "seconds_since_review" not in ann_existing:
        cursor.execute("ALTER TABLE annotations ADD COLUMN seconds_since_review INTEGER")
        logger.info("Added column annotations.seconds_since_review")

    conn.commit()
    conn.close()


def main():
    # Ensure tables exist
    Base.metadata.create_all(engine)
    # Add any new columns to existing tables
    try:
        _migrate_add_columns()
    except Exception:
        logger.warning("Column migration failed (table may not exist yet)", exc_info=True)

    # On startup, recover any submissions stuck from a previous crash
    recover_stuck_submissions()

    logger.info("Worker started. Polling every %ds...", settings.worker_poll_interval)
    last_cleanup = time.time()
    while True:
        # Run cleanup every 10 minutes
        if time.time() - last_cleanup > 600:
            cleanup_old_submissions()
            last_cleanup = time.time()

        submission = get_next_pending()
        if submission:
            logger.info("[%s] Processing submission: %s (mode=%s)", submission.key, submission.filename, submission.mode.value)
            try:
                process_submission(submission)
            except Exception:
                tb = traceback.format_exc()
                logger.error("[%s] Processing failed:\n%s", submission.key, tb)
                if _is_budget_error(tb):
                    # Budget error — mark as failed with a recognizable tag so the
                    # frontend can show a friendly message, then pause before retrying.
                    logger.warning("[%s] Budget exceeded — marking as budget_exhausted.", submission.key)
                    update_status(
                        submission.key,
                        SubmissionStatus.failed,
                        error="[OUT_OF_BUDGET] The service is temporarily out of API credit. "
                              "Your submission has been saved and will be retried automatically once credit is restored.",
                    )
                else:
                    update_status(submission.key, SubmissionStatus.failed, error=tb[-500:])
        else:
            time.sleep(settings.worker_poll_interval)


if __name__ == "__main__":
    main()
