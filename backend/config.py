import json

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Deprecated: server-funded OCR now routes through the LiteLLM proxy.
    # Kept so existing .env files with MISTRAL_API_KEY still validate.
    mistral_api_key: str = ""

    # Tavily
    tavily_api_key: str = ""

    # LiteLLM — used for the review agent and the server-funded OCR (routes
    # through the proxy)
    litellm_api_key: str = ""
    litellm_base_url: str = "https://cmu.litellm.ai"

    # OCR
    # Server-funded (queue mode): routes through the LiteLLM proxy above.
    ocr_model: str = "azure_ai/mistral-document-ai-2512"
    # BYOK mode: submitter's own Mistral key against the public Mistral API.
    mistral_base_url: str = "https://api.mistral.ai"
    byok_ocr_model: str = "mistral-ocr-latest"

    # Admin
    admin_api_key: str = ""

    # Email / SMTP
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = "noreply@cmu-paper-reviewer.com"

    # Database
    database_url: str = "sqlite+aiosqlite:///./data/reviewer.db"

    # Data directory
    data_dir: str = "./data"

    # CORS. Localhost dev origins are always allowed. Production origins (the
    # deployed frontend, backend host, custom domains) are supplied per-
    # deployment via the CORS_ORIGINS env var as a comma-separated list, e.g.
    #   CORS_ORIGINS=https://prometheus-eval.github.io,https://your-domain.org
    # so deployment-specific hosts stay out of the source tree.
    cors_origins_extra: str = Field("", alias="CORS_ORIGINS")

    @property
    def cors_origins(self) -> list[str]:
        defaults = [
            "http://localhost:5500",
            "http://localhost:8000",
            "http://localhost:3000",
            "http://127.0.0.1:5500",
            "http://127.0.0.1:8000",
        ]
        raw = self.cors_origins_extra.strip()
        if not raw:
            extra = []
        elif raw.startswith("["):
            # Backwards-compatible: older .env files store a JSON array.
            extra = [str(o).strip() for o in json.loads(raw)]
        else:
            extra = [o.strip() for o in raw.split(",") if o.strip()]
        # De-duplicate while preserving order (env may repeat a localhost default).
        seen = set()
        return [o for o in defaults + extra if o and not (o in seen or seen.add(o))]

    # Rate limiting
    max_submissions_per_ip_per_day: int = 3

    # Submission size limits
    max_manuscript_pages: int = 100         # main PDF page cap
    max_supplementary_pages: int = 50        # supplementary PDF page cap
    max_pdf_mb: int = 50                     # per-PDF upload size cap
    max_code_zip_mb: int = 50                # code .zip upload size cap
    max_code_uncompressed_mb: int = 500      # anti-zip-bomb: total uncompressed
    max_code_files: int = 10000              # anti-zip-bomb: entry count

    # OCR: the Azure mistral-document-ai deployment on the LiteLLM proxy
    # rejects documents over 30 pages, so larger PDFs are split into chunks
    # of this many pages and OCR'd sequentially, then stitched back together.
    ocr_max_pages_per_request: int = 30
    # Chunks are OCR'd concurrently (each call is network-bound). Capped to
    # bound peak memory, since each in-flight chunk holds its page images'
    # base64 in memory.
    ocr_max_concurrent_chunks: int = 3

    # Worker
    worker_poll_interval: int = 10
    review_models: list[str] = [
        "litellm_proxy/azure_ai/gpt-5.5",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
