"""Background worker that polls SQLite for pending submissions and processes them.

Run as: python -m backend.worker
"""

import asyncio
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


def process_submission(submission: Submission):
    key = submission.key
    pdf_file = upload_path(key, submission.filename)
    is_byok = submission.mode == SubmissionMode.byok

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

        # Step 2: Review
        logger.info("[%s] Starting review...", key)
        update_status(key, SubmissionStatus.reviewing)
        reviewer = ReviewService(
            litellm_api_key=submission.user_litellm_api_key if is_byok else None,
            litellm_base_url=submission.user_litellm_base_url if is_byok else None,
            tavily_api_key=submission.user_tavily_api_key if is_byok else None,
        )
        reviewer.run_review(key)
        logger.info("[%s] Review complete.", key)

        # Step 3: Generate PDF
        logger.info("[%s] Generating PDF...", key)
        generate_review_pdf(key, model_name=settings.review_model)
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


CLEANUP_MAX_AGE = timedelta(hours=1)


def cleanup_old_submissions():
    """Delete submissions and their files older than CLEANUP_MAX_AGE.

    Submissions that have annotations are preserved permanently.
    """
    cutoff = datetime.now(timezone.utc) - CLEANUP_MAX_AGE
    with SessionLocal() as session:
        # Find keys that have annotations — these are preserved permanently
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
                logger.info("[%s] Skipping cleanup — submission has annotations.", sub.key)
                continue
            # Remove review directory
            review_d = review_dir(sub.key)
            if review_d.exists():
                shutil.rmtree(review_d, ignore_errors=True)
            # Remove uploaded file
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


def main():
    # Ensure tables exist
    Base.metadata.create_all(engine)

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
                update_status(submission.key, SubmissionStatus.failed, error=tb[-500:])
        else:
            time.sleep(settings.worker_poll_interval)


if __name__ == "__main__":
    main()
