"""
User-configurable parameters for PeerReview Bench evaluation.

All parameters can be set via CLI args, environment variables, or by
instantiating EvalConfig directly in a script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

_BENCH_DIR = Path(__file__).resolve().parent.parent


@dataclass
class EvalConfig:
    """Central configuration for the evaluation pipeline."""

    # ---- Paper preparation ----
    paper_root: Path = _BENCH_DIR / "papers"
    """Root directory for paper{N}/ subdirectories. Created by prepare_papers.py."""

    # ---- Review generation ----
    model_name: str = "litellm_proxy/anthropic/claude-opus-4-6"
    """LLM model name for the OpenHands agent reviewer."""

    max_items: int = 5
    """Maximum number of review items the agent should produce per paper."""

    criteria_preset: str = "nature"
    """Evaluation criteria preset: 'nature' or 'neurips'."""

    litellm_api_key: Optional[str] = None
    """API key for LiteLLM proxy. Falls back to LITELLM_API_KEY env var."""

    litellm_base_url: str = "https://cmu.litellm.ai"
    """LiteLLM proxy base URL."""

    tavily_api_key: Optional[str] = None
    """Tavily API key for literature search. Falls back to TAVILY_API_KEY env var."""

    max_iterations: int = 5000
    """Max OpenHands iterations per paper conversation."""

    # ---- Review parsing ----
    review_format: str = "peerreview_bench"
    """Review format to parse: 'peerreview_bench' (our structured format) or
    'byoj' (user provides review_items.json directly)."""

    # ---- Recall evaluation ----
    similarity_model: str = "litellm_proxy/anthropic/claude-opus-4-6"
    """LLM judge model for 4-way similarity classification (recall metric)."""

    similarity_concurrency: int = 16
    """Concurrent similarity judge calls."""

    # ---- Precision evaluation ----
    meta_review_mode: str = "agent"
    """Meta-reviewer mode: 'agent' (OpenHands agent) or 'llm' (direct LLM call)."""

    meta_review_model: str = "litellm_proxy/anthropic/claude-opus-4-6"
    """Model for meta-review judge (precision metric)."""

    meta_review_concurrency: int = 16
    """Concurrent meta-review calls (only for 'llm' mode)."""

    # ---- General ----
    output_dir: Path = _BENCH_DIR / "outputs" / "evaluation"
    """Directory for evaluation outputs."""

    limit: Optional[int] = None
    """Only evaluate the first N papers (smoke testing)."""

    paper_ids: Optional[List[int]] = None
    """Specific paper IDs to evaluate. If None, evaluate all."""

    skip_existing: bool = True
    """Skip papers that already have a generated review."""
