from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Mistral OCR
    mistral_api_key: str = ""

    # Tavily
    tavily_api_key: str = ""

    # LiteLLM — used for the review agent (routes to Claude via proxy)
    litellm_api_key: str = ""
    litellm_base_url: str = "https://cmu.litellm.ai"

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

    # CORS
    cors_origins: list[str] = [
        "http://localhost:5500",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "http://127.0.0.1:8000",
        "https://prometheus-eval.github.io",
        "http://35.231.39.20",
    ]

    # Worker
    worker_poll_interval: int = 10
    review_model: str = "litellm_proxy/neulab/claude-opus-4-5-20251101"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
