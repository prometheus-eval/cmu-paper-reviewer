import datetime

from pydantic import BaseModel, EmailStr

from backend.models import SubmissionStatus


class SubmitResponse(BaseModel):
    key: str
    message: str
    mode: str


class StatusResponse(BaseModel):
    key: str
    status: SubmissionStatus
    filename: str
    mode: str
    created_at: datetime.datetime
    error_message: str | None = None


class ReviewResponse(BaseModel):
    key: str
    status: SubmissionStatus
    review_markdown: str | None = None


class ProgressEvent(BaseModel):
    step: int
    timestamp: str
    tool_name: str | None = None
    summary: str | None = None


class ProgressResponse(BaseModel):
    total_steps: int
    last_action_summary: str | None = None
    events: list[ProgressEvent] = []


class VerificationCodeFile(BaseModel):
    name: str
    size: int


class VerificationCodeResponse(BaseModel):
    files: list[VerificationCodeFile] = []


class AnnotationRequest(BaseModel):
    item_number: int
    annotator_id: str = "anonymous"
    correctness: str | None = None
    significance: str | None = None
    evidence_quality: str | None = None


class AnnotationResponse(BaseModel):
    key: str
    item_number: int
    annotator_id: str = "anonymous"
    correctness: str | None = None
    significance: str | None = None
    evidence_quality: str | None = None
