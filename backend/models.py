import datetime
import enum
import secrets
import string

from sqlalchemy import Boolean, DateTime, Enum, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class SubmissionMode(str, enum.Enum):
    queue = "queue"
    byok = "byok"


class SubmissionStatus(str, enum.Enum):
    pending = "pending"
    ocr = "ocr"
    reviewing = "reviewing"
    completed = "completed"
    failed = "failed"


def generate_key(length: int = 12) -> str:
    alphabet = string.ascii_lowercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


class Submission(Base):
    __tablename__ = "submissions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(12), unique=True, index=True, default=generate_key)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    filename: Mapped[str] = mapped_column(String(255))
    mode: Mapped[SubmissionMode] = mapped_column(
        Enum(SubmissionMode), default=SubmissionMode.queue
    )
    status: Mapped[SubmissionStatus] = mapped_column(
        Enum(SubmissionStatus), default=SubmissionStatus.pending
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    has_code: Mapped[bool] = mapped_column(Boolean, default=False)
    has_supplementary: Mapped[bool] = mapped_column(Boolean, default=False)
    user_mistral_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_litellm_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_litellm_base_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    user_tavily_api_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(12), index=True)
    item_number: Mapped[int] = mapped_column(Integer)
    annotator_id: Mapped[str] = mapped_column(String(36), default="anonymous")
    correctness: Mapped[str | None] = mapped_column(String(20), nullable=True)
    significance: Mapped[str | None] = mapped_column(String(30), nullable=True)
    evidence_quality: Mapped[str | None] = mapped_column(String(20), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
