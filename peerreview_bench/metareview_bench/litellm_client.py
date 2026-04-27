"""
Thin wrapper around `litellm.completion` that points at the CMU LiteLLM proxy
and exposes a single `call_llm(...)` entry point for the meta-reviewer predictors.

All meta-reviewer models are routed via the CMU LiteLLM proxy. The base URL
and API key are configurable via environment variables or arguments.

Environment variables (read at import time):
  LITELLM_BASE_URL   — default "https://cmu.litellm.ai"
  LITELLM_API_KEY    — API key for the proxy
  LITELLM_KEY        — alias for LITELLM_API_KEY
  LITELLM_KEY_FILE   — path to a file whose single line is the API key
                       (fallback if LITELLM_API_KEY / LITELLM_KEY not set)

If none of the above are set, we fall back to reading the key from
`<peerreview_bench>/api_key/litellm.txt` (relative to this file). When read
from disk, we also populate `os.environ["LITELLM_API_KEY"]` and
`os.environ["LITELLM_KEY"]` so downstream callers (openhands, litellm itself)
can pick it up without re-reading the file.

Typical usage:

    from litellm_client import call_llm
    text = call_llm(
        model="litellm_proxy/azure_ai/gpt-5.4",
        messages=[{"role": "user", "content": "Hi"}],
        response_format={"type": "json_object"},
    )

We import `litellm` lazily so importing this module doesn't pull in the full
LiteLLM SDK unless the caller actually invokes `call_llm`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "https://cmu.litellm.ai"
# peerreview_bench/api_key/litellm.txt (relative to this file)
_DEFAULT_KEY_FILE = Path(__file__).resolve().parent.parent / "api_key" / "litellm.txt"
_DEFAULT_BASE_URL_FILE = Path(__file__).resolve().parent.parent / "api_key" / "base_url.txt"


def _resolve_api_key() -> str:
    """Read the API key from env var or key file.

    Resolution order:
      1. $LITELLM_API_KEY
      2. $LITELLM_KEY
      3. $LITELLM_KEY_FILE (path)
      4. peerreview_bench/api_key/litellm.txt
      5. ./key_litellm.txt (legacy)

    When resolved from a file, we also set LITELLM_API_KEY and LITELLM_KEY in
    the process environment so downstream SDKs see it without us passing it
    through every call.
    """
    key = os.environ.get("LITELLM_API_KEY") or os.environ.get("LITELLM_KEY")
    if key:
        return key.strip()

    candidates: List[Path] = []
    env_key_file = os.environ.get("LITELLM_KEY_FILE")
    if env_key_file:
        candidates.append(Path(env_key_file))
    candidates.append(_DEFAULT_KEY_FILE)
    candidates.append(Path("./key_litellm.txt"))

    for p in candidates:
        if p.is_file():
            key = p.read_text(encoding="utf-8").strip()
            if key:
                os.environ["LITELLM_API_KEY"] = key
                os.environ["LITELLM_KEY"] = key
                return key

    raise RuntimeError(
        "Could not find a LiteLLM API key. Set LITELLM_API_KEY in your "
        "environment, set LITELLM_KEY_FILE to a file path, or put the key "
        f"in {_DEFAULT_KEY_FILE}."
    )


def _resolve_base_url() -> str:
    url = os.environ.get("LITELLM_BASE_URL")
    if url:
        return url.rstrip("/")
    if _DEFAULT_BASE_URL_FILE.is_file():
        url = _DEFAULT_BASE_URL_FILE.read_text(encoding="utf-8").strip()
        if url:
            return url.rstrip("/")
    return DEFAULT_BASE_URL


def call_llm(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    response_format: Optional[Dict[str, Any]] = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    timeout: int = 120,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """Call a model through the LiteLLM proxy and return the assistant text.

    Args:
        model: the model id. Should be prefixed with `litellm_proxy/`
            (e.g. `"litellm_proxy/azure_ai/gpt-5.4"`). If the user forgets the
            prefix, we auto-add it.
        messages: OpenAI-style messages list (supports multimodal content
            blocks for models that accept image_url parts).
        response_format: passed through to LiteLLM's `response_format` kwarg.
            Examples:
              - `{"type": "json_object"}`
              - `{"type": "json_schema", "json_schema": {...}}`
        max_tokens / temperature / timeout: standard completion params.
        api_key, base_url: override the default env-var-resolved ones.
        extra_kwargs: any additional kwargs forwarded to `litellm.completion`.

    Returns:
        The assistant's text (concatenation of all text blocks in the first
        choice's message).
    """
    import litellm  # Lazy import: only pull in the big SDK when actually used

    if not model.startswith("litellm_proxy/"):
        model = "litellm_proxy/" + model

    api_key = api_key or _resolve_api_key()
    base_url = base_url or _resolve_base_url()

    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "timeout": timeout,
        "api_key": api_key,
        "api_base": base_url,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    response = litellm.completion(**kwargs)
    msg = response.choices[0].message
    content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
    if not content:
        return ""
    if isinstance(content, str):
        return content
    # Some models return a list of content blocks; concatenate text blocks
    out: List[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                out.append(block.get("text", ""))
            elif "text" in block:
                out.append(str(block.get("text", "")))
        else:
            out.append(str(block))
    return "".join(out)


def call_embedding(
    model: str,
    texts: List[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    dimensions: Optional[int] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> List[List[float]]:
    """Call an embedding model via the LiteLLM proxy.

    Args:
        model: model id (will be prefixed with `litellm_proxy/` if missing).
        texts: list of input strings.
        dimensions: optional truncation / target dimensionality.

    Returns:
        A list of embedding vectors (each is a list[float]).
    """
    import litellm

    if not model.startswith("litellm_proxy/"):
        model = "litellm_proxy/" + model

    api_key = api_key or _resolve_api_key()
    base_url = base_url or _resolve_base_url()

    kwargs: Dict[str, Any] = {
        "model": model,
        "input": texts,
        "api_key": api_key,
        "api_base": base_url,
    }
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    response = litellm.embedding(**kwargs)
    # LiteLLM normalizes responses to OpenAI shape: response.data is a list of
    # dicts each with an 'embedding' field.
    out: List[List[float]] = []
    for item in response.data:
        emb = item["embedding"] if isinstance(item, dict) else item.embedding
        out.append(list(emb))
    return out
