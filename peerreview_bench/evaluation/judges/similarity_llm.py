"""Self-contained LLM-call + answer-parsing helpers for the recall judge.

Vendored from similarity_check/expert_annotation_similarity/baselines/llm_classifier.py.
The API key / base URL are read from the environment (LITELLM_API_KEY,
LITELLM_BASE_URL); set them before running the evaluation.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

from tqdm import tqdm

DEFAULT_BASE_URL = "https://cmu.litellm.ai"


def _resolve_api_key() -> str:
    key = os.environ.get("LITELLM_API_KEY") or os.environ.get("LITELLM_KEY")
    if key:
        return key.strip()
    raise RuntimeError(
        "Could not find a LiteLLM API key. Set LITELLM_API_KEY in your environment."
    )


def _resolve_base_url() -> str:
    url = os.environ.get("LITELLM_BASE_URL")
    if url:
        return url.rstrip("/")
    return DEFAULT_BASE_URL


ANSWER_RE = re.compile(
    r'<answer>\s*(.*?)\s*</answer>',
    re.IGNORECASE | re.DOTALL,
)

_FOURWAY_LABELS = {
    'same subject, same argument, same evidence',
    'same subject, same argument, different evidence',
    'same subject, different argument',
    'different subject',
}

def extract_4way_answer(text: str) -> Optional[str]:
    """Return one of the four long-form labels, or None if the response
    can't be parsed into any of them.

    Tolerates:
      - multiple <answer> tags (returns the last one)
      - surrounding whitespace, trailing punctuation
      - wrapping quotes, backticks, or asterisks (models sometimes add
        markdown formatting despite the prompt saying not to)
      - trailing-period variants of the canonical labels
      - missing closing tag (falls back to everything after the last
        opening `<answer>` if no closed tag is found)
    """
    if not text:
        return None

    candidates: List[str] = list(ANSWER_RE.findall(text))

    # Fallback: if the model wrote `<answer> ...` but forgot the closing
    # tag, grab everything from the LAST opening tag to end-of-text.
    if not candidates:
        open_matches = list(re.finditer(r'<answer\s*>', text, re.IGNORECASE))
        if open_matches:
            last_open = open_matches[-1].end()
            candidates.append(text[last_open:].strip())

    if not candidates:
        return None

    raw = candidates[-1]

    # Strip common wrapping characters the prompt told the model NOT to add
    # but which they sometimes add anyway.
    raw = raw.strip()
    for _ in range(3):
        stripped = raw.strip('`').strip('"').strip("'").strip('*').strip()
        if stripped == raw:
            break
        raw = stripped

    normalized = ' '.join(raw.lower().strip().rstrip('.').split())
    if normalized in _FOURWAY_LABELS:
        return normalized
    for canonical in _FOURWAY_LABELS:
        if normalized == canonical.rstrip('.'):
            return canonical

    # Last-ditch fallback: did the normalized string UNIQUELY contain one
    # of the four label strings as a substring? (Handles cases like
    # "the answer is: same subject, same argument, same evidence")
    contained = [label for label in _FOURWAY_LABELS if label in normalized]
    if len(contained) == 1:
        return contained[0]
    return None

_ANTHROPIC_RESPONSE_HEADROOM = 4096

_LLM_RETRY_BACKOFFS = [10, 30, 60, 120, 240]

def bare_name_from_model(model: str) -> str:
    """Strip the `litellm_proxy/` prefix for human-readable log messages."""
    return model[len('litellm_proxy/'):] if model.startswith('litellm_proxy/') else model

def build_reasoning_kwargs(model: str, max_tokens: int) -> Dict[str, Any]:
    """Return the `extra_kwargs` blob to enable max-effort reasoning/thinking
    for the given model, sized against its `max_tokens` output budget."""
    bare = model[len('litellm_proxy/'):] if model.startswith('litellm_proxy/') else model
    kwargs: Dict[str, Any] = {'reasoning_effort': 'high'}
    if bare.startswith('anthropic/'):
        budget = max(1024, max_tokens - _ANTHROPIC_RESPONSE_HEADROOM)
        kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': budget}
    return kwargs

def _extract_reasoning_content(response_obj: Any) -> Optional[str]:
    """Pull standardized reasoning content from a LiteLLM response.

    LiteLLM exposes this as `response.choices[0].message.reasoning_content`
    across providers. Some providers also return Anthropic-style
    `thinking_blocks`. We take whichever we can find.
    """
    try:
        choice = response_obj.choices[0]
        msg = choice.message
    except (AttributeError, IndexError):
        return None

    reasoning = getattr(msg, 'reasoning_content', None)
    if reasoning:
        return reasoning

    thinking = getattr(msg, 'thinking_blocks', None)
    if thinking:
        parts: List[str] = []
        for b in thinking:
            if isinstance(b, dict):
                parts.append(b.get('thinking') or b.get('text') or '')
            else:
                parts.append(str(getattr(b, 'thinking', '') or getattr(b, 'text', '')))
        return '\n\n'.join(p for p in parts if p)

    return None

def _call_llm_with_reasoning(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    max_tokens: int,
    temperature: float,
    extra_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Wrapper around litellm.completion that additionally returns
    reasoning_content. We intentionally sidestep litellm_client.call_llm
    because that helper flattens the response to plain text and drops the
    reasoning field.

    Temperature handling: Anthropic extended thinking REQUIRES temperature=1.
    Gemini 3 prefers 1.0 for reasoning quality. Azure GPT-5 accepts any
    value (defaults to 1.0). We default to 1.0 across the board and handle
    'param not supported' errors by retrying once without the temperature
    kwarg, since some LiteLLM routing paths still strip temperature on
    reasoning models.
    """
    import litellm  # lazy import

    # Reuse the api_key / base_url resolution from litellm_client by doing
    # a trivial throwaway call — actually, just inline the resolution:

    # Only enforce litellm_proxy/ prefix when using the CMU proxy
    base_url = _resolve_base_url()
    if 'cmu.litellm.ai' in base_url and not model.startswith('litellm_proxy/'):
        raise ValueError(
            f"CMU LiteLLM proxy requires 'litellm_proxy/' prefix in model name. "
            f"Got: '{model}'. Use e.g. 'litellm_proxy/azure_ai/gpt-5.4'."
        )

    kwargs: Dict[str, Any] = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'api_key': _resolve_api_key(),
        'api_base': base_url,
        'timeout': 600,
        'drop_params': True,  # silently drop unsupported params (e.g., reasoning_effort for newer models)
    }
    kwargs.update(extra_kwargs)

    def _is_retryable_error(exc: Exception) -> bool:
        """True for any transient error that should be retried with backoff.

        Includes:
          - 429 / rate-limit (provider TPM/RPM cap)
          - 5xx / internal server errors (proxy or upstream hiccup)
          - connection / network errors (dropped sockets, DNS blips)
          - timeouts

        Empirically the dominant failure mode on longer-running thinking
        calls (Claude Opus 4.6 especially) is the proxy returning
        'InternalServerError: Connection error', not a 429.
        """
        m = str(exc).lower()
        cls_name = type(exc).__name__
        return (
            # rate limit
            cls_name == 'RateLimitError'
            or '429' in m
            or 'resource_exhausted' in m
            or 'resource exhausted' in m
            or ('rate' in m and 'limit' in m)
            or 'too many requests' in m
            # server errors
            or cls_name in ('InternalServerError', 'ServiceUnavailableError',
                            'APIConnectionError', 'APITimeoutError',
                            'Timeout', 'TimeoutError', 'ConnectionError',
                            'ReadTimeoutError', 'ReadTimeout')
            or any(code in m for code in (' 500', ' 502', ' 503', ' 504'))
            or 'internal server error' in m
            or 'service unavailable' in m
            or 'bad gateway' in m
            or 'gateway timeout' in m
            or 'connection error' in m
            or 'connection reset' in m
            or 'connection aborted' in m
            or 'connection refused' in m
            or 'timed out' in m
            or 'read timed out' in m
        )

    # Outer loop: retry on rate-limit (429) with exponential backoff. We
    # sleep between attempts; the total worst-case wait is ~8 min across
    # 5 retries before giving up. Concurrent workers amplify the chance
    # of tripping a provider's TPM cap, so the patient backoff is worth
    # the extra latency on the affected calls.
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
            # Special-case: temperature param rejected — retry once
            # without it and continue with the same attempt count.
            if 'temperature' in msg and (
                'unsupported' in msg or 'not support' in msg
                or 'does not support' in msg or 'invalid' in msg
            ):
                kwargs.pop('temperature', None)
                try:
                    response = litellm.completion(**kwargs)
                    break
                except Exception as e2:
                    last_err = e2
                    if not _is_retryable_error(e2):
                        raise
            elif ('reasoning_effort' in msg or 'thinking' in msg) and (
                'unsupported' in msg or 'not support' in msg
                or 'does not support' in msg
            ):
                # Proxy rejects reasoning params for newer models —
                # retry without them (model may still reason internally)
                kwargs.pop('reasoning_effort', None)
                kwargs.pop('thinking', None)
                try:
                    response = litellm.completion(**kwargs)
                    break
                except Exception as e2:
                    last_err = e2
                    if not _is_retryable_error(e2):
                        raise
            elif not _is_retryable_error(e):
                raise
            # transient error (429 / 5xx / connection / timeout): back off
            if attempt < len(_LLM_RETRY_BACKOFFS):
                tqdm.write(
                    f'  transient error from {bare_name_from_model(model)}: '
                    f'{type(e).__name__}; '
                    f'sleeping {_LLM_RETRY_BACKOFFS[attempt]}s before '
                    f'retry {attempt+1}/{len(_LLM_RETRY_BACKOFFS)}...'
                )
    if response is None:
        assert last_err is not None
        raise last_err

    choice = response.choices[0]
    msg = choice.message
    content = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
    if content is None:
        text = ''
    elif isinstance(content, str):
        text = content
    else:
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    parts.append(block.get('text', ''))
                elif 'text' in block:
                    parts.append(str(block.get('text', '')))
            else:
                parts.append(str(block))
        text = ''.join(parts)

    return {
        'content': text,
        'reasoning_content': _extract_reasoning_content(response),
    }
