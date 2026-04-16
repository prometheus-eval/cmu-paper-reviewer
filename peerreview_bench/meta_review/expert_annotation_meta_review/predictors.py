"""
Pluggable meta-reviewer predictors for the expert_annotation_meta_review
benchmark.

Two prompt modes are supported per predictor:
    - "axis"     — LLM emits per-axis cascade labels (correctness,
                   significance, evidence)
    - "tenclass" — LLM predicts one of the 10 collapsed class labels
                   that a pair of meta-reviewers would jointly produce

Predictor kinds:
    RandomPredictor     — uniform random (mode-aware)
    MajorityPredictor   — always predicts the most-common class
    ConstantPredictor   — always predicts a caller-provided dict
    LiteLLMMetaReviewer — thinking-mode LLM judge with retries, code+image
                          attachments, and multi-strategy JSON parsing

The LiteLLMMetaReviewer reuses the same retry / thinking-mode machinery
that lives in similarity_check/expert_annotation_similarity/baselines/
llm_classifier.py — the code is copy-pasted rather than imported so the
running similarity job (which reads from that sibling file) is never
disturbed while this file is being edited.
"""

from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------- Path plumbing for the shared top-level modules ---------------
# Layout:
#   peerreview_bench/meta_review/litellm_client.py                <- _META_REVIEW_DIR
#   peerreview_bench/meta_review/model_config.py                  <- _META_REVIEW_DIR
#   peerreview_bench/meta_review/image_mapping.py                 <- _META_REVIEW_DIR
#   peerreview_bench/meta_review/expert_annotation_meta_review/predictors.py <- _HERE
#   peerreview_bench/load_data.py                                  <- _BENCH_DIR
_HERE = Path(__file__).resolve().parent
_META_REVIEW_DIR = _HERE.parent
_BENCH_DIR = _META_REVIEW_DIR.parent

for p in (_HERE, _META_REVIEW_DIR, _BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from prompts import (  # noqa: E402
    AXIS_JSON_SCHEMA,
    TENCLASS_JSON_SCHEMA,
    TENCLASS_LABELS,
    TENCLASS_LABEL_TO_ID,
    build_messages,
    extract_axis_answer,
    extract_tenclass_answer,
    axis_to_tenclass_label,
)

# Top-level shared modules (keep at meta_review/ root so the similarity job
# still finds them).
from model_config import (  # noqa: E402
    get_response_format_mode,
    supports_multimodal,
    get_max_output_tokens,
)
from image_mapping import (  # noqa: E402
    find_figure_references,
    select_images_for_review_item,
)


# ============================================================================
# Interface
# ============================================================================

class MetaReviewPredictor(ABC):
    name: str = "unknown"
    mode: str = "axis"  # "axis" or "tenclass"

    @abstractmethod
    def predict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Return a prediction dict for one meta_reviewer / expert_annotation row.

        For mode == 'axis', the return value must contain the keys:
            reasoning, correctness, significance, evidence
        For mode == 'tenclass', the return value must contain:
            reasoning, label, label_id
        """
        raise NotImplementedError


# ============================================================================
# Baseline predictors (no LLM calls)
# ============================================================================

_DEFAULT_AXIS = {
    "reasoning": "(baseline)",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
}

_DEFAULT_TENCLASS = {
    "reasoning": "(baseline)",
    "label": "correct_significant_sufficient",
    "label_id": 1,
}


def _default_for_mode(mode: str) -> Dict[str, Any]:
    return dict(_DEFAULT_AXIS) if mode == "axis" else dict(_DEFAULT_TENCLASS)


class ConstantPredictor(MetaReviewPredictor):
    """Always returns the same prediction (mode-aware default)."""

    name = "constant"

    def __init__(
        self,
        prediction: Optional[Dict[str, Any]] = None,
        *,
        mode: str = "axis",
        **_,
    ):
        self.mode = mode
        self._pred = prediction or _default_for_mode(mode)

    def predict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return dict(self._pred)


class MajorityPredictor(ConstantPredictor):
    name = "majority"


class RandomPredictor(MetaReviewPredictor):
    name = "random"

    def __init__(self, seed: int = 0, *, mode: str = "axis", **_):
        self.mode = mode
        self._rng = random.Random(seed)

    def predict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == "tenclass":
            lab = self._rng.choice(TENCLASS_LABELS)
            return {
                "reasoning": "(random)",
                "label": lab,
                "label_id": TENCLASS_LABEL_TO_ID[lab],
            }
        # axis mode — cascade-respecting random
        corr = self._rng.choice(["Correct", "Not Correct"])
        if corr == "Not Correct":
            return {
                "reasoning": "(random)",
                "correctness": corr,
                "significance": None,
                "evidence": None,
            }
        sig = self._rng.choice(
            ["Not Significant", "Marginally Significant", "Significant"]
        )
        if sig == "Not Significant":
            return {
                "reasoning": "(random)",
                "correctness": corr,
                "significance": sig,
                "evidence": None,
            }
        evi = self._rng.choice(["Sufficient", "Requires More"])
        return {
            "reasoning": "(random)",
            "correctness": corr,
            "significance": sig,
            "evidence": evi,
        }


# ============================================================================
# Code / text artifact selection (copy-pasted + adapted from similarity_check)
# ============================================================================
#
# Same priority rules, same byte budgets. The review-item version below
# takes a single review text instead of the two-item pair version.

_CODE_EXTS = (
    'py', 'ipynb', 'js', 'ts', 'jsx', 'tsx', 'cpp', 'hpp', 'cc',
    'cs', 'java', 'go', 'rs', 'sh', 'rb', 'scala', 'kt', 'mm', 'lua',
    'swift', 'dart', 'sql', 'yaml', 'yml', 'toml', 'md', 'rst', 'tex',
)
_FILENAME_RE = re.compile(
    r"\b([\w./-]+\.(?:" + '|'.join(_CODE_EXTS) + r"))\b",
    re.IGNORECASE,
)

_ARTIFACT_MAX_FILE_BYTES = 50_000
_ARTIFACT_MAX_TOTAL_BYTES = 400_000

_SKIP_ARTIFACT_BASENAMES = {
    "preprint.md",  # already inlined in paper_content
}

_HIGH_VALUE_BASENAMES = {
    "readme.md", "readme.rst", "readme.txt",
    "setup.py", "setup.cfg", "pyproject.toml",
    "requirements.txt", "environment.yml", "environment.yaml",
    "dockerfile",
    "main.py", "train.py", "run.py",
}

_SOURCE_CODE_EXTENSIONS = {
    "py", "ipynb", "js", "ts", "jsx", "tsx",
    "cpp", "cc", "cxx", "hpp", "c", "h",
    "java", "scala", "kt", "go", "rs", "rb", "cs",
    "swift", "dart", "m", "mm", "lua", "r", "jl",
    "pl", "sh", "fish", "zsh", "bash",
    "sql", "proto",
}


def find_file_references(text: str) -> List[str]:
    """Return distinct filenames mentioned in `text`."""
    if not text:
        return []
    out: List[str] = []
    seen = set()
    for m in _FILENAME_RE.finditer(text):
        name = m.group(1)
        key = name.lower()
        if key not in seen:
            seen.add(key)
            out.append(name)
    return out


def _decode_text_blob(blob: Any, max_bytes: int) -> Optional[str]:
    if blob is None:
        return None
    if isinstance(blob, dict):
        is_text = blob.get("is_text")
        content = blob.get("content_bytes")
    else:
        is_text = None
        content = blob
    if is_text is False:
        return None
    if not isinstance(content, (bytes, bytearray)):
        return None
    try:
        text = bytes(content).decode("utf-8", errors="replace")
    except Exception:
        return None
    if len(text) > max_bytes:
        text = text[:max_bytes] + "\n\n[... file truncated ...]\n"
    return text


def _blob_size(blob: Any) -> int:
    if isinstance(blob, dict):
        sz = blob.get("size_bytes")
        if isinstance(sz, int):
            return sz
        content = blob.get("content_bytes")
    else:
        content = blob
    if isinstance(content, (bytes, bytearray)):
        return len(content)
    return 0


def select_review_item_artifacts(
    review_text: str,
    file_refs: List[Dict[str, Any]],
    hash_to_bytes: Dict[str, Any],
    *,
    max_file_bytes: int = _ARTIFACT_MAX_FILE_BYTES,
    max_total_bytes: int = _ARTIFACT_MAX_TOTAL_BYTES,
) -> List[Dict[str, Any]]:
    """Bulk-with-budget artifact ingestion for a single review item.

    Priority:
      1. Files explicitly named in the review item text
      2. High-value config/entrypoint files (README, setup.py, ...)
      3. Source code files (.py, .ipynb, .cpp, ...)
      4. Other text files

    Returns [{path, content}] dicts. Empty if no text file_refs.
    """
    if not file_refs:
        return []

    review_named = {n.lower().lstrip("./") for n in find_file_references(review_text or "")}

    candidates: List[Tuple[Dict[str, Any], str, int]] = []
    for ref in file_refs:
        path = ref.get("path") or ""
        if not path:
            continue
        basename = path.rsplit("/", 1)[-1].lower()
        if basename in _SKIP_ARTIFACT_BASENAMES:
            continue
        blob = hash_to_bytes.get(ref.get("content_hash"))
        if blob is None:
            continue
        if isinstance(blob, dict) and blob.get("is_text") is False:
            continue
        size = _blob_size(blob)
        if size == 0:
            continue
        candidates.append((ref, basename, size))

    if not candidates:
        return []

    def _priority(rbs: Tuple[Dict[str, Any], str, int]) -> int:
        ref, basename, _ = rbs
        path_lower = ref["path"].lower().lstrip("./")
        for name in review_named:
            if (
                path_lower == name
                or path_lower.endswith("/" + name)
                or basename == name.rsplit("/", 1)[-1]
            ):
                return 0
        if basename in _HIGH_VALUE_BASENAMES:
            return 1
        ext = basename.rsplit(".", 1)[-1] if "." in basename else ""
        if ext in _SOURCE_CODE_EXTENSIONS:
            return 2
        return 3

    candidates.sort(key=lambda rbs: (_priority(rbs), rbs[2]))

    attached: List[Dict[str, Any]] = []
    attached_paths: set = set()
    total_bytes = 0
    for ref, _basename, _size in candidates:
        path = ref["path"]
        if path in attached_paths:
            continue
        if total_bytes >= max_total_bytes:
            break
        blob = hash_to_bytes.get(ref.get("content_hash"))
        text = _decode_text_blob(blob, max_bytes=max_file_bytes)
        if text is None:
            continue
        remaining = max_total_bytes - total_bytes
        if len(text) > remaining:
            text = text[:remaining] + "\n\n[... artifact budget exhausted ...]\n"
        attached.append({"path": path, "content": text})
        attached_paths.add(path)
        total_bytes += len(text)

    return attached


# ============================================================================
# LLM call machinery — thinking mode + retries + reasoning extraction
# ============================================================================

_ANTHROPIC_RESPONSE_HEADROOM = 4096
_LLM_RETRY_BACKOFFS = [10, 30, 60, 120, 240]


def bare_name_from_model(model: str) -> str:
    return model[len("litellm_proxy/"):] if model.startswith("litellm_proxy/") else model


def _slug(model: str) -> str:
    bare = bare_name_from_model(model)
    return bare.replace("/", "__").replace(":", "__").replace(".", "_")


def build_reasoning_kwargs(model: str, max_tokens: int) -> Dict[str, Any]:
    """Return the `extra_kwargs` blob to enable max-effort reasoning/thinking
    for the given model, sized against its `max_tokens` output budget.

    Provider-specific handling:
      - Anthropic: explicit `thinking` block + `reasoning_effort="high"`
      - Gemini: `reasoning_effort="high"` (LiteLLM maps to thinking_level)
      - Azure AI GPT-5: `reasoning_effort="high"` (native OpenAI reasoning)
      - Azure AI Grok / Kimi: LiteLLM's AzureAIStudioConfig does NOT
        include `reasoning_effort` in supported params — passing it
        triggers UnsupportedParamsError. These models have built-in
        reasoning anyway, so we skip it.
      - Fireworks AI (qwen3p6-plus): thinking is always-on;
        no kwarg needed.
    """
    bare = bare_name_from_model(model)

    # Fireworks AI models have mandatory always-on chain-of-thought.
    if bare.startswith("fireworks_ai/"):
        return {}

    # Azure AI models routed through AzureAIStudioConfig don't support
    # reasoning_effort (only OpenAI o-series does). GPT-5 on Azure works
    # because it's routed through a different path. For Grok and Kimi,
    # skip reasoning_effort — they have built-in reasoning anyway.
    _AZURE_AI_SKIP_REASONING = {
        "azure_ai/grok-4-1-fast-reasoning",
        "azure_ai/Kimi-K2.5",
    }
    if bare in _AZURE_AI_SKIP_REASONING:
        return {}

    kwargs: Dict[str, Any] = {"reasoning_effort": "high"}
    if bare.startswith("anthropic/"):
        budget = max(1024, max_tokens - _ANTHROPIC_RESPONSE_HEADROOM)
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
    return kwargs


def _extract_reasoning_content(response_obj: Any) -> Optional[str]:
    """Pull standardized reasoning content from a LiteLLM response."""
    try:
        choice = response_obj.choices[0]
        msg = choice.message
    except (AttributeError, IndexError):
        return None

    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning:
        return reasoning

    thinking = getattr(msg, "thinking_blocks", None)
    if thinking:
        parts: List[str] = []
        for b in thinking:
            if isinstance(b, dict):
                parts.append(b.get("thinking") or b.get("text") or "")
            else:
                parts.append(str(getattr(b, "thinking", "") or getattr(b, "text", "")))
        return "\n\n".join(p for p in parts if p)

    return None


def _is_retryable_error(exc: Exception) -> bool:
    """True for any transient error that should be retried with backoff."""
    m = str(exc).lower()
    cls_name = type(exc).__name__
    return (
        cls_name == "RateLimitError"
        or "429" in m
        or "resource_exhausted" in m
        or "resource exhausted" in m
        or ("rate" in m and "limit" in m)
        or "too many requests" in m
        or cls_name in (
            "InternalServerError", "ServiceUnavailableError",
            "APIConnectionError", "APITimeoutError",
            "Timeout", "TimeoutError", "ConnectionError",
            "ReadTimeoutError", "ReadTimeout",
        )
        or any(code in m for code in (" 500", " 502", " 503", " 504"))
        or "internal server error" in m
        or "service unavailable" in m
        or "bad gateway" in m
        or "gateway timeout" in m
        or "connection error" in m
        or "connection reset" in m
        or "connection aborted" in m
        or "connection refused" in m
        or "timed out" in m
        or "read timed out" in m
    )


def _call_llm_with_reasoning(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    extra_kwargs: Dict[str, Any],
    response_format: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Wrapper around litellm.completion that additionally returns
    reasoning_content. Handles transient error retries with exponential
    backoff.
    """
    import litellm  # lazy import

    if not model.startswith("litellm_proxy/"):
        model = "litellm_proxy/" + model

    from litellm_client import _resolve_api_key, _resolve_base_url

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "api_key": _resolve_api_key(),
        "api_base": _resolve_base_url(),
        "timeout": 600,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    kwargs.update(extra_kwargs)

    response = None
    last_err: Optional[Exception] = None
    for attempt, wait in enumerate([0] + _LLM_RETRY_BACKOFFS):
        if wait:
            time.sleep(wait)
        try:
            response = litellm.completion(**kwargs)
            break
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            if "temperature" in msg and (
                "unsupported" in msg or "not support" in msg
                or "does not support" in msg or "invalid" in msg
            ):
                kwargs.pop("temperature", None)
                try:
                    response = litellm.completion(**kwargs)
                    break
                except Exception as e2:
                    last_err = e2
                    if not _is_retryable_error(e2):
                        raise
            elif "response_format" in msg and (
                "unsupported" in msg or "not support" in msg or "invalid" in msg
            ):
                # Provider doesn't accept our response_format — retry without it
                kwargs.pop("response_format", None)
                try:
                    response = litellm.completion(**kwargs)
                    break
                except Exception as e2:
                    last_err = e2
                    if not _is_retryable_error(e2):
                        raise
            elif not _is_retryable_error(e):
                raise
            if attempt < len(_LLM_RETRY_BACKOFFS):
                print(
                    f"  transient error from {bare_name_from_model(model)}: "
                    f"{type(e).__name__}; "
                    f"sleeping {_LLM_RETRY_BACKOFFS[attempt]}s before "
                    f"retry {attempt+1}/{len(_LLM_RETRY_BACKOFFS)}...",
                    flush=True,
                )
    if response is None:
        assert last_err is not None
        raise last_err

    choice = response.choices[0]
    msg = choice.message
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    if content is None:
        text = ""
    elif isinstance(content, str):
        text = content
    else:
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif "text" in block:
                    parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        text = "".join(parts)

    return {
        "content": text,
        "reasoning_content": _extract_reasoning_content(response),
    }


# ============================================================================
# LiteLLM predictor — the main entrypoint for LLM-backed meta-review
# ============================================================================

class LiteLLMMetaReviewer(MetaReviewPredictor):
    """Thinking-mode LLM meta-reviewer with retries, image and code-artifact
    attachment, and multi-strategy JSON parsing.

    Args:
        model: LiteLLM model id, with or without the litellm_proxy/ prefix.
        mode: 'axis' or 'tenclass' — selects the prompt family and parser.
        attach_images: override multimodal detection. Defaults to whatever
            supports_multimodal() says for the given model.
        attach_artifacts: if True, attach text/code files from file_refs
            that are mentioned by name in the review item. Defaults True.
        hash_to_bytes_loader: callable returning a content_hash->blob map.
            Used by both image and artifact selection. Lazily invoked.
        max_images: cap images per item. Default None (no cap).
        max_tokens: completion budget. Default = model's catalog
            max_output_tokens from model_config.py.
        temperature: default 1.0 (required by Anthropic extended thinking,
            recommended for Gemini 3 reasoning).
    """

    def __init__(
        self,
        model: str,
        *,
        mode: str,
        attach_images: Optional[bool] = None,
        attach_artifacts: bool = True,
        hash_to_bytes_loader=None,
        max_images: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 1.0,
        artifact_max_file_bytes: int = _ARTIFACT_MAX_FILE_BYTES,
        artifact_max_total_bytes: int = _ARTIFACT_MAX_TOTAL_BYTES,
    ):
        if mode not in ("axis", "tenclass"):
            raise ValueError(f"mode must be 'axis' or 'tenclass', got {mode!r}")
        self.model = model
        self.mode = mode
        bare = bare_name_from_model(model)
        self._bare_model = bare
        self.name = f"litellm_{bare.replace('/', '_').replace('.', '_')}_{mode}"

        self._response_format_mode = get_response_format_mode(bare)
        if attach_images is None:
            attach_images = supports_multimodal(bare)
        self.attach_images = attach_images
        self.attach_artifacts = attach_artifacts
        self.max_images = max_images

        self.max_tokens = (
            max_tokens if max_tokens is not None else get_max_output_tokens(bare)
        )
        self.temperature = temperature
        self.reasoning_kwargs = build_reasoning_kwargs(model, self.max_tokens)

        self.artifact_max_file_bytes = artifact_max_file_bytes
        self.artifact_max_total_bytes = artifact_max_total_bytes

        self._hash_to_bytes_loader = hash_to_bytes_loader
        self._hash_to_bytes: Optional[Dict[str, Any]] = None

    def _get_hash_to_bytes(self) -> Optional[Dict[str, Any]]:
        need = self.attach_images or self.attach_artifacts
        if not need:
            return None
        if self._hash_to_bytes is None and self._hash_to_bytes_loader is not None:
            self._hash_to_bytes = self._hash_to_bytes_loader()
        return self._hash_to_bytes

    def _build_response_format(self) -> Optional[Dict[str, Any]]:
        if self._response_format_mode == "json_schema":
            schema = AXIS_JSON_SCHEMA if self.mode == "axis" else TENCLASS_JSON_SCHEMA
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "meta_review_prediction" if self.mode == "axis" else "meta_review_tenclass",
                    "schema": schema,
                    "strict": True,
                },
            }
        if self._response_format_mode == "json_object":
            return {"type": "json_object"}
        return None

    def _gather_review_text_for_matching(self, row: Dict[str, Any]) -> str:
        """Concatenate the review parts that should feed into image/artifact
        selection (figure reference detection etc.)."""
        parts = [
            row.get("review_content") or row.get("review_item") or "",
            row.get("review_claim") or "",
            row.get("review_evidence") or "",
        ]
        return "\n".join(p for p in parts if p)

    def predict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        # ---- Image and artifact selection ----
        images: List[Dict[str, Any]] = []
        artifacts: List[Dict[str, Any]] = []
        hash_to_bytes = self._get_hash_to_bytes()
        file_refs = row.get("file_refs") or []
        review_text = self._gather_review_text_for_matching(row)

        if self.attach_images and hash_to_bytes and file_refs:
            if find_figure_references(review_text):
                images = select_images_for_review_item(
                    review_text,
                    file_refs,
                    hash_to_bytes,
                    max_images=self.max_images,
                )

        if self.attach_artifacts and hash_to_bytes and file_refs:
            artifacts = select_review_item_artifacts(
                review_text,
                file_refs,
                hash_to_bytes,
                max_file_bytes=self.artifact_max_file_bytes,
                max_total_bytes=self.artifact_max_total_bytes,
            )

        # ---- Build messages ----
        review_content = row.get("review_content") or row.get("review_item") or ""
        messages = build_messages(
            mode=self.mode,
            paper_title=row.get("paper_title") or "",
            paper_content=row.get("paper_content") or "",
            review_content=review_content,
            item_number=row.get("item_number") or row.get("review_item_number") or 0,
            review_claim=row.get("review_claim"),
            review_evidence=row.get("review_evidence"),
            review_cited_references=row.get("review_cited_references"),
            images=images if images else None,
            artifacts=artifacts if artifacts else None,
        )

        response_format = self._build_response_format()

        # ---- Call the LLM ----
        t0 = time.time()
        try:
            call_result = _call_llm_with_reasoning(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_kwargs=self.reasoning_kwargs,
                response_format=response_format,
            )
        except Exception as e:
            out = _default_for_mode(self.mode)
            out["reasoning"] = f"(litellm error: {type(e).__name__}: {e})"
            out["_parsed"] = False
            out["_error"] = f"{type(e).__name__}: {e}"
            out["_elapsed_seconds"] = time.time() - t0
            out["_n_images"] = len(images)
            out["_n_artifacts"] = len(artifacts)
            return out

        elapsed = time.time() - t0
        response_text = call_result.get("content") or ""
        reasoning_content = call_result.get("reasoning_content")

        # ---- Parse ----
        if self.mode == "axis":
            parsed = extract_axis_answer(response_text)
        else:
            parsed = extract_tenclass_answer(response_text)

        if parsed is None:
            out = _default_for_mode(self.mode)
            out["reasoning"] = f"(failed to parse: {response_text[:200]})"
            out["_parsed"] = False
            out["_error"] = None
            out["_response"] = response_text
            out["_reasoning_content"] = reasoning_content
            out["_elapsed_seconds"] = elapsed
            out["_n_images"] = len(images)
            out["_n_artifacts"] = len(artifacts)
            return out

        parsed["_parsed"] = True
        parsed["_error"] = None
        parsed["_elapsed_seconds"] = elapsed
        parsed["_n_images"] = len(images)
        parsed["_n_artifacts"] = len(artifacts)
        # Stash reasoning_content separately (not part of the prediction
        # dict itself; the runner pulls it out for the output record)
        parsed["_reasoning_content"] = reasoning_content

        # Enforce cascade constraints for axis mode (strict nullification)
        if self.mode == "axis":
            if parsed["correctness"] == "Not Correct":
                parsed["significance"] = None
                parsed["evidence"] = None
            elif parsed["significance"] == "Not Significant":
                parsed["evidence"] = None

        return parsed


# ============================================================================
# Registry / factory
# ============================================================================

BASELINE_REGISTRY = {
    "random": RandomPredictor,
    "majority": MajorityPredictor,
    "constant": ConstantPredictor,
}


def get_predictor(
    name_or_model: str,
    *,
    mode: str = "axis",
    **kwargs,
) -> MetaReviewPredictor:
    """Factory: pick a baseline by short name, or treat the arg as a
    LiteLLM model id. `mode` selects the prompt family (axis/tenclass)."""
    key = name_or_model.lower()
    if key in BASELINE_REGISTRY:
        return BASELINE_REGISTRY[key](mode=mode, **kwargs)
    return LiteLLMMetaReviewer(model=name_or_model, mode=mode, **kwargs)
