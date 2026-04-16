"""
LLM-as-a-judge 4-way similarity baseline for the similarity_check eval set.

For each of the 238 pairs, we ask a frontier LLM to classify the
relationship between two review items into one of four long-form labels:

  - "same subject, same argument, same evidence"       (near-paraphrase)
  - "same subject, same argument, different evidence"  (convergent conclusion)
  - "same subject, different argument"                 (topical neighbor)
  - "different subject"                                 (unrelated)

This matches the `finegrained_label` column of the HF schema. The binary
label (similar / not_similar) is derived from the fine-grained prediction
via prompts.fourway_to_binary.

Design choices (requested by the user):

1. **Routing**: every model goes through the CMU LiteLLM proxy using the
   shared `litellm_client.call_llm` from ../meta_review. The same three
   litellm_proxy ids are supported — Azure GPT-5.4, Gemini 3.1 Pro
   Preview, Anthropic Claude Opus 4.6. New models only need to appear in
   meta_review/model_config.py.

2. **Thinking mode**: `reasoning_effort="high"` is passed universally (the
   LiteLLM docs show it mapping to the appropriate per-provider knob —
   Gemini `thinking_level=high`, Azure GPT-5 reasoning, Anthropic adaptive
   thinking). In addition, for Anthropic we pass
   `thinking={"type": "enabled", "budget_tokens": ...}` as a top-level
   kwarg (more explicit than the reasoning_effort fallback).

3. **No truncation**: we feed the full `paper_content` into every prompt.
   All three requested models have >=1M-token context — the longest paper
   in the dataset (~106K chars ≈ ~26K tokens) easily fits.

4. **Max output**: the completion `max_tokens` defaults to the per-model
   max_output_tokens from meta_review/model_config.py (so we never cap the
   model below its catalog limit).

5. **Multimodal**: if EITHER review item mentions a figure/table reference
   (regex from meta_review/image_mapping.find_figure_references), we
   attach the matching images from the paper's `images_list.json`, looked
   up through the `reviewer` HF config's file_refs + the `submitted_papers`
   HF config's hash->bytes map. Pairs where neither review mentions a
   figure are sent as text-only.

Outputs land in `--output-dir` as `llm_judge_4way_<model_slug>.json`:

    {
      "metadata": {model, n_pairs, n_parsed, parse_rate, accuracy_4way,
                   accuracy_binary},
      "results": [
        {eval_pair_id, binary_label, finegrained_label, pair_type,
         parsed_answer, parsed_binary, reasoning_content, response,
         image_paths, elapsed_seconds},
        ...
      ],
    }
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# --- HF Hub network timeouts --------------------------------------------
# Bump the default 10s HEAD-request timeout so `submitted_papers` (which
# is a ~2 GB multi-shard config) doesn't fail on slow connections. Apply
# BEFORE importing huggingface_hub (the env vars are snapshotted at import).
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

# --- Path plumbing: make sibling modules importable ------------------------
# Insert in REVERSE priority order (sys.path.insert(0,...) bumps each one to
# the top). Final priority: expert_annotation_similarity > bench > meta_review
# — so `from prompts import ...` resolves to our prompts.py (which sits in
# expert_annotation_similarity/) rather than meta_review/prompts.py (they're
# different files with the same name).
# Path layout after reorg:
#   peerreview_bench/similarity_check/expert_annotation_similarity/baselines/llm_classifier.py
#   peerreview_bench/similarity_check/expert_annotation_similarity/prompts.py  <- _EA_DIR
#   peerreview_bench/meta_review/                                               <- _META_REVIEW_DIR
#   peerreview_bench/load_data.py                                               <- _BENCH_DIR
_HERE = Path(__file__).resolve().parent              # .../expert_annotation_similarity/baselines
_EA_DIR = _HERE.parent                                 # .../expert_annotation_similarity
_SIMCHECK_DIR = _EA_DIR.parent                         # .../similarity_check
_BENCH_DIR = _SIMCHECK_DIR.parent                      # .../peerreview_bench
_META_REVIEW_DIR = _BENCH_DIR / 'meta_review'

for p in (_META_REVIEW_DIR, _BENCH_DIR, _EA_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# similarity_check
from load_eval_set import (  # noqa: E402
    load_similarity_eval_set,
    load_paper_file_refs,
    SimilarityPair,
)
from prompts import (  # noqa: E402
    FOURWAY_SYSTEM_PROMPT,
    FOURWAY_USER_PROMPT_TEMPLATE,
    fourway_to_binary,
)

# meta_review infra we reuse
from litellm_client import call_llm  # noqa: E402
from model_config import (  # noqa: E402
    get_max_output_tokens,
    supports_multimodal,
)
from image_mapping import (  # noqa: E402
    find_figure_references,
    select_images_for_review_item,
)

# load submitted_papers lazily
from load_data import load_submitted_papers  # noqa: E402


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# File-reference detection (code / text artifacts)
# ---------------------------------------------------------------------------
#
# When a review item explicitly names a file in the paper's supplementary
# code/data bundle (e.g. "the bug in `trainer_ppi.py`", "Figure 2 is
# generated by MLmodels.ipynb"), we want the LLM judge to see the actual
# file content so it can verify whether the two items are pointing at the
# same code-level issue. Empirically, 44 of 238 pairs in the eval set (~18%)
# name at least one code file, and ignoring those files means the LLM has
# to guess whether two reviews are talking about the same function.
#
# Detection strategy: a regex over common code / text extensions. We
# intentionally exclude single-letter extensions (`c`, `h`, `m`, `r`)
# because review text frequently contains initials like "W.H. Smith" or
# "M.C. Jones" that would false-positive. Papers that use pure C headers
# are rare in this dataset so the miss rate is low.

_CODE_EXTS = (
    'py', 'ipynb', 'js', 'ts', 'jsx', 'tsx', 'cpp', 'hpp', 'cc',
    'cs', 'java', 'go', 'rs', 'sh', 'rb', 'scala', 'kt', 'mm', 'lua',
    'swift', 'dart', 'sql', 'yaml', 'yml', 'toml', 'md', 'rst', 'tex',
)
_FILENAME_RE = re.compile(
    r'\b([\w./-]+\.(?:' + '|'.join(_CODE_EXTS) + r'))\b',
    re.IGNORECASE,
)


def find_file_references(text: str) -> List[str]:
    """Return distinct filenames (basenames or relative paths) mentioned in
    the given text. Matches only files whose extension is in _CODE_EXTS."""
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


# Artifact ingestion budgets. We want to feed the LLM as much of the
# paper's supplementary text as fits, with a bounded prompt size.
#
# - max_file_bytes caps any single file (so one giant notebook doesn't eat
#   the whole budget).
# - max_total_bytes caps the entire artifact section per pair. At 400K
#   chars ≈ 100K tokens, plus ~30K tokens of paper content and ~128K of
#   reserved output, we stay comfortably under the 1M-token budget shared
#   by gpt-5.4 / gemini-3.1-pro / claude-opus-4-6.
_ARTIFACT_MAX_FILE_BYTES = 50_000    # ~12K tokens per file, hard cap
_ARTIFACT_MAX_TOTAL_BYTES = 400_000  # ~100K tokens across all attached files


# Files from file_refs whose *basename* should NEVER be attached as an
# artifact — typically because they're already inlined into the prompt
# via pair.paper_content (the HF schema's `paper_content` column carries
# `preprint/preprint.md` verbatim). Basename-only comparison keeps this
# robust against directory-layout differences across papers.
_SKIP_ARTIFACT_BASENAMES = {
    'preprint.md',
}


# High-value config / entrypoint files that get priority over "generic"
# source code when filling the budget. Lowercase basenames.
_HIGH_VALUE_BASENAMES = {
    'readme.md', 'readme.rst', 'readme.txt',
    'setup.py', 'setup.cfg', 'pyproject.toml',
    'requirements.txt', 'environment.yml', 'environment.yaml',
    'dockerfile',
    'main.py', 'train.py', 'run.py',
}

# Source-code file extensions (lowercase, no dot). Files with these
# extensions beat non-source text files when filling the budget — reviewers
# almost always care about the algorithm code, not Makefiles or YAML.
_SOURCE_CODE_EXTENSIONS = {
    'py', 'ipynb', 'js', 'ts', 'jsx', 'tsx',
    'cpp', 'cc', 'cxx', 'hpp', 'c', 'h',
    'java', 'scala', 'kt', 'go', 'rs', 'rb', 'cs',
    'swift', 'dart', 'm', 'mm', 'lua', 'r', 'jl',
    'pl', 'sh', 'fish', 'zsh', 'bash',
    'sql', 'proto',
}


def _decode_text_blob(blob: Any, max_bytes: int) -> Optional[str]:
    """Extract utf-8 text from a submitted_papers hash_to_bytes entry.
    Returns None for binary or unreadable blobs. Truncates at max_bytes."""
    if blob is None:
        return None
    if isinstance(blob, dict):
        is_text = blob.get('is_text')
        content = blob.get('content_bytes')
    else:
        is_text = None
        content = blob
    if is_text is False:
        return None
    if not isinstance(content, (bytes, bytearray)):
        return None
    try:
        text = bytes(content).decode('utf-8', errors='replace')
    except Exception:
        return None
    if len(text) > max_bytes:
        text = text[:max_bytes] + '\n\n[... file truncated ...]\n'
    return text


def _blob_size(blob: Any) -> int:
    """Best-effort byte size for a hash_to_bytes entry (without decoding)."""
    if isinstance(blob, dict):
        sz = blob.get('size_bytes')
        if isinstance(sz, int):
            return sz
        content = blob.get('content_bytes')
    else:
        content = blob
    if isinstance(content, (bytes, bytearray)):
        return len(content)
    return 0


def select_pair_artifacts(
    pair: SimilarityPair,
    file_refs: List[Dict[str, Any]],
    hash_to_bytes: Dict[str, Any],
    *,
    max_file_bytes: int = _ARTIFACT_MAX_FILE_BYTES,
    max_total_bytes: int = _ARTIFACT_MAX_TOTAL_BYTES,
) -> List[Dict[str, Any]]:
    """Bulk-with-budget artifact ingestion.

    For the given pair, resolve every TEXT file in the paper's `file_refs`
    through `hash_to_bytes` and return as many as fit within
    `max_total_bytes`. Files explicitly named in either review get
    priority-1 slots so they're never dropped in favor of unrelated code.

    Ordering when the paper's total text exceeds the budget:
      1. Files whose name appears in item_a.text or item_b.text (regex
         detection via find_file_references)
      2. High-value config/entrypoint files (README, setup.py, …)
      3. Source code files (.py, .ipynb, .cpp, .java, …) — the algorithm
         code reviewers actually discuss. Smallest-first within the tier.
      4. Other text files (Makefile, .yaml, .md, …) — smallest-first.

    Exclusions:
      - Binary blobs (skipped via is_text check)
      - Basenames in _SKIP_ARTIFACT_BASENAMES (e.g. preprint.md — already
        inlined into pair.paper_content by the HF schema)

    Returns a list of {path, content} dicts. Empty if the paper has no
    text file_refs at all.
    """
    if not file_refs:
        return []

    combined = f'{pair.item_a.text}\n\n{pair.item_b.text}'
    review_named = {n.lower().lstrip('./') for n in find_file_references(combined)}

    # Pre-filter: keep only entries with a path + a resolvable text blob.
    # Record basename and size for downstream ordering without decoding
    # every file (decoding is expensive for large blobs).
    candidates: List[Tuple[Dict[str, Any], str, int]] = []  # (ref, basename, size)
    for ref in file_refs:
        path = ref.get('path') or ''
        if not path:
            continue
        basename = path.rsplit('/', 1)[-1].lower()
        if basename in _SKIP_ARTIFACT_BASENAMES:
            continue  # already inlined as pair.paper_content
        blob = hash_to_bytes.get(ref.get('content_hash'))
        if blob is None:
            continue
        # Skip binary-typed blobs without decoding.
        if isinstance(blob, dict) and blob.get('is_text') is False:
            continue
        size = _blob_size(blob)
        if size == 0:
            continue
        candidates.append((ref, basename, size))

    if not candidates:
        return []

    # Priority classification.
    def _priority(ref_basename_size: Tuple[Dict[str, Any], str, int]) -> int:
        ref, basename, _ = ref_basename_size
        path_lower = ref['path'].lower().lstrip('./')
        # Priority 0: explicitly named in the reviews. Match by exact path,
        # path-suffix (/name), or basename so "setup.py" in a review resolves
        # to "code/pkg/setup.py" in file_refs.
        for name in review_named:
            if (path_lower == name
                    or path_lower.endswith('/' + name)
                    or basename == name.rsplit('/', 1)[-1]):
                return 0
        if basename in _HIGH_VALUE_BASENAMES:
            return 1
        # Priority 2: actual source code (.py, .ipynb, .cpp, .java, ...).
        # Reviewers almost always care about the algorithm implementation,
        # not build config or YAML.
        ext = basename.rsplit('.', 1)[-1] if '.' in basename else ''
        if ext in _SOURCE_CODE_EXTENSIONS:
            return 2
        return 3

    # Sort by (priority asc, size asc). Size-asc on ties gives "breadth
    # before depth": we pack many small files before one large one.
    candidates.sort(key=lambda rbs: (_priority(rbs), rbs[2]))

    attached: List[Dict[str, Any]] = []
    attached_paths: set = set()
    total_bytes = 0

    for ref, _basename, _size in candidates:
        path = ref['path']
        if path in attached_paths:
            continue
        if total_bytes >= max_total_bytes:
            break
        blob = hash_to_bytes.get(ref.get('content_hash'))
        text = _decode_text_blob(blob, max_bytes=max_file_bytes)
        if text is None:
            continue
        # If this file alone would blow past the total budget, truncate
        # further so we can still include a meaningful prefix.
        remaining = max_total_bytes - total_bytes
        if len(text) > remaining:
            text = text[:remaining] + '\n\n[... artifact budget exhausted ...]\n'
        attached.append({'path': path, 'content': text})
        attached_paths.add(path)
        total_bytes += len(text)

    return attached


# ---------------------------------------------------------------------------
# Message + image building
# ---------------------------------------------------------------------------

def build_user_text(
    pair: SimilarityPair,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> str:
    base = FOURWAY_USER_PROMPT_TEMPLATE.format(
        paper_text=pair.paper_content,
        reviewer_a=pair.item_a.reviewer_id,
        reviewer_b=pair.item_b.reviewer_id,
        item_a=pair.item_a.text,
        item_b=pair.item_b.text,
    )
    if not artifacts:
        return base

    # Append referenced file contents as fenced code blocks AFTER the
    # paper text so the model sees them as "additional supplementary files
    # that the reviewers called out by name."
    parts = [base, '\n\n---\n\n### Referenced files (mentioned by name in one or both review items)\n']
    for a in artifacts:
        parts.append(f'\n#### `{a["path"]}`\n```\n{a["content"]}\n```\n')
    return ''.join(parts)


def select_pair_images(
    pair: SimilarityPair,
    file_refs: List[Dict[str, Any]],
    hash_to_bytes: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Attach images iff at least one of the two review items mentions a
    figure / table reference. Returns an empty list otherwise.

    Uses meta_review/image_mapping.find_figure_references to detect
    mentions, and then select_images_for_review_item to resolve the
    matching images through the paper's images_list.json.
    """
    combined = f'{pair.item_a.text}\n\n{pair.item_b.text}'
    refs = find_figure_references(combined)
    if not refs:
        return []
    return select_images_for_review_item(
        combined,
        file_refs,
        hash_to_bytes,
        max_images=None,
    )


def build_messages(
    pair: SimilarityPair,
    images: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build an OpenAI-style chat messages list with optional image and
    text-artifact content. Text artifacts (code/text files referenced by
    name in either review item) are inlined into the user text. Images
    are attached as multimodal image_url blocks."""
    user_text = build_user_text(pair, artifacts=artifacts)

    if images:
        import base64
        content: List[Dict[str, Any]] = [{'type': 'text', 'text': user_text}]
        for img in images:
            img_bytes = img.get('bytes')
            if not img_bytes:
                continue
            mime = img.get('mime') or 'image/jpeg'
            b64 = base64.b64encode(img_bytes).decode('ascii')
            content.append({
                'type': 'image_url',
                'image_url': {'url': f'data:{mime};base64,{b64}'},
            })
            caption = img.get('caption', '')
            if caption:
                content.append({
                    'type': 'text',
                    'text': f"(Caption for {img.get('img_path', 'image')}): {caption}",
                })
        user_msg: Dict[str, Any] = {'role': 'user', 'content': content}
    else:
        user_msg = {'role': 'user', 'content': user_text}

    return [
        {'role': 'system', 'content': FOURWAY_SYSTEM_PROMPT},
        user_msg,
    ]


# ---------------------------------------------------------------------------
# Reasoning / thinking configuration
# ---------------------------------------------------------------------------
#
# Different providers budget thinking differently:
#
#   - Anthropic Claude Opus 4.6 : thinking tokens + visible response share the
#     same `max_tokens` bucket (128K total). `thinking.budget_tokens` must be
#     strictly less than max_tokens. Setting it to (max_tokens - headroom)
#     gives us the largest practical thinking budget. The explicit
#     `thinking: {type: "enabled", budget_tokens: N}` form is deprecated on
#     4.6 (replaced by `type: "adaptive"` + output_config.effort) but is
#     still functional, and it lets us be EXPLICIT about requesting the
#     model's maximum thinking allowance.
#
#   - Gemini 3.1 Pro Preview    : thinking tokens come out of the 65,536
#     output budget. `thinking_level="high"` is the categorical max (the
#     four levels are minimal / low / medium / high). LiteLLM maps
#     `reasoning_effort="high"` → `thinking_level="high"`, so that's already
#     the max effort — there's no explicit token knob for us to tune.
#
#   - Azure GPT-5.4             : reasoning tokens count against
#     max_completion_tokens. `reasoning_effort="high"` is the max effort
#     setting (low / medium / high). Again, no explicit token knob.
#
# So: every model gets `reasoning_effort="high"`, and Anthropic additionally
# gets an explicit `thinking` block sized to (max_tokens - 4096). 4K of
# headroom is plenty — our visible response is a step-by-step chain of
# thought plus a single `<answer>...</answer>` tag, typically ~500-2000
# tokens total.

_ANTHROPIC_RESPONSE_HEADROOM = 4096

# Retry schedule for transient LLM errors: rate-limit (429 /
# RESOURCE_EXHAUSTED), 5xx server errors, connection drops, timeouts.
# With concurrent workers we're more likely to trip provider TPM caps
# and proxy-level connection instabilities, so the wrapper backs off
# patiently before giving up.
_LLM_RETRY_BACKOFFS = [10, 30, 60, 120, 240]


def build_reasoning_kwargs(model: str, max_tokens: int) -> Dict[str, Any]:
    """Return the `extra_kwargs` blob to enable max-effort reasoning/thinking
    for the given model, sized against its `max_tokens` output budget."""
    bare = model[len('litellm_proxy/'):] if model.startswith('litellm_proxy/') else model
    kwargs: Dict[str, Any] = {'reasoning_effort': 'high'}
    if bare.startswith('anthropic/'):
        budget = max(1024, max_tokens - _ANTHROPIC_RESPONSE_HEADROOM)
        kwargs['thinking'] = {'type': 'enabled', 'budget_tokens': budget}
    return kwargs


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

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

    if not model.startswith('litellm_proxy/'):
        model = 'litellm_proxy/' + model

    # Reuse the api_key / base_url resolution from litellm_client by doing
    # a trivial throwaway call — actually, just inline the resolution:
    from litellm_client import _resolve_api_key, _resolve_base_url

    kwargs: Dict[str, Any] = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'api_key': _resolve_api_key(),
        'api_base': _resolve_base_url(),
        'timeout': 600,
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


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _slug(model: str) -> str:
    bare = model[len('litellm_proxy/'):] if model.startswith('litellm_proxy/') else model
    return bare.replace('/', '__').replace(':', '__').replace('.', '_')


def bare_name_from_model(model: str) -> str:
    """Strip the `litellm_proxy/` prefix for human-readable log messages."""
    return model[len('litellm_proxy/'):] if model.startswith('litellm_proxy/') else model


def _process_pair(
    pair: SimilarityPair,
    *,
    model: str,
    effective_max_tokens: int,
    temperature: float,
    reasoning_kwargs: Dict[str, Any],
    attach_images: bool,
    paper_file_refs: Dict[int, List[Dict[str, Any]]],
    hash_to_bytes: Dict[str, Any],
    artifact_max_file_bytes: int,
    artifact_max_total_bytes: int,
) -> Dict[str, Any]:
    """Per-pair worker. Selects images/artifacts, calls the LLM, parses the
    answer, and returns a result dict. Safe to call concurrently from
    multiple threads — reads only from immutable shared state.
    """
    # -------- Image + code-artifact selection --------
    images: List[Dict[str, Any]] = []
    image_paths: List[str] = []
    artifacts: List[Dict[str, Any]] = []
    artifact_paths: List[str] = []
    artifact_total_bytes = 0

    if attach_images:
        file_refs = paper_file_refs.get(pair.paper_id, [])
        if file_refs:
            # Images only if figure/table actually mentioned.
            if find_figure_references(
                f'{pair.item_a.text}\n\n{pair.item_b.text}'
            ):
                images = select_pair_images(pair, file_refs, hash_to_bytes)
                image_paths = [img.get('img_path', '') for img in images]

            # Artifacts: bulk — always run, returns [] if paper has
            # no resolvable text files.
            artifacts = select_pair_artifacts(
                pair, file_refs, hash_to_bytes,
                max_file_bytes=artifact_max_file_bytes,
                max_total_bytes=artifact_max_total_bytes,
            )
            artifact_paths = [a['path'] for a in artifacts]
            artifact_total_bytes = sum(len(a['content']) for a in artifacts)

    messages = build_messages(
        pair,
        images=images if images else None,
        artifacts=artifacts if artifacts else None,
    )

    # -------- LLM call (with rate-limit retry inside) --------
    t0 = time.time()
    response_text = ''
    reasoning_content = None
    error: Optional[str] = None
    try:
        call_result = _call_llm_with_reasoning(
            model=model,
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            extra_kwargs=reasoning_kwargs,
        )
        response_text = call_result['content']
        reasoning_content = call_result['reasoning_content']
    except Exception as e:
        error = f'{type(e).__name__}: {e}'
    elapsed = time.time() - t0

    # -------- Parse --------
    parsed = extract_4way_answer(response_text) if response_text else None
    parsed_binary: Optional[str] = None
    if parsed is not None:
        parsed_binary = fourway_to_binary(parsed)

    return {
        'eval_pair_id': pair.eval_pair_id,
        'paper_id': pair.paper_id,
        'binary_label': pair.binary_label,
        'finegrained_label': pair.finegrained_label,
        'pair_type': pair.pair_type,
        'parsed_answer': parsed,
        'parsed_binary': parsed_binary,
        'response': response_text,
        'reasoning_content': reasoning_content,
        'image_paths': image_paths,
        'n_images': len(image_paths),
        'artifact_paths': artifact_paths,
        'n_artifacts': len(artifact_paths),
        'artifact_total_bytes': artifact_total_bytes,
        'elapsed_seconds': elapsed,
        'error': error,
    }


def run(
    model: str,
    output_dir: Path,
    limit: Optional[int],
    *,
    max_tokens: Optional[int],
    temperature: float,
    no_images: bool,
    artifact_max_file_bytes: int,
    artifact_max_total_bytes: int,
    progress_every: int,
    concurrency: int,
) -> None:
    pairs = load_similarity_eval_set()
    print(f'Loaded {len(pairs)} pairs')
    if limit:
        pairs = pairs[:limit]
        print(f'  limited to first {limit}')

    bare_model = model[len('litellm_proxy/'):] if model.startswith('litellm_proxy/') else model
    effective_max_tokens = max_tokens if max_tokens is not None else get_max_output_tokens(bare_model)
    reasoning_kwargs = build_reasoning_kwargs(model, effective_max_tokens)
    print(f'Model:             {model}')
    print(f'Max output tokens: {effective_max_tokens}')
    print(f'Temperature:       {temperature}')
    print(f'Reasoning kwargs:  {reasoning_kwargs}')
    print(f'Concurrency:       {concurrency} worker(s)')

    attach_images = (not no_images) and supports_multimodal(bare_model)
    print(f'Attach images:     {attach_images}')
    print(f'Artifact budget:   {artifact_max_total_bytes} bytes total, '
          f'{artifact_max_file_bytes} bytes/file')

    # ---- Eagerly load file_refs + submitted_papers so workers can read
    # them without locks. (Lazy-loading would race across threads.)
    paper_file_refs: Dict[int, List[Dict[str, Any]]] = {}
    hash_to_bytes: Dict[str, Any] = {}

    if attach_images:
        print('Loading file_refs from reviewer config (for images_list.json lookup)...')
        paper_file_refs = load_paper_file_refs()
        print(f'  loaded file_refs for {len(paper_file_refs)} papers')

        # 5-attempt backoff for submitted_papers cold-cache download.
        _BACKOFFS = [30, 60, 120, 240, 300]
        print('Loading submitted_papers config for image bytes...')
        last_err = None
        for attempt, wait in enumerate(_BACKOFFS):
            try:
                hash_to_bytes = load_submitted_papers()
                print(f'  loaded {len(hash_to_bytes)} file blobs')
                last_err = None
                break
            except Exception as e:
                last_err = e
                print(f'  attempt {attempt+1}/{len(_BACKOFFS)} failed '
                      f'({type(e).__name__}: {e}); retrying in {wait}s...',
                      flush=True)
                if attempt < len(_BACKOFFS) - 1:
                    time.sleep(wait)
        if last_err is not None:
            print(f'  FAILED to load submitted_papers after '
                  f'{len(_BACKOFFS)} attempts ({type(last_err).__name__}: '
                  f'{last_err}).')
            print('  Falling back to text-only for the entire run.')
            attach_images = False
            hash_to_bytes = {}

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f'llm_judge_4way_{_slug(model)}.json'
    print(f'Writing to {out_path}')

    # Pre-allocate results so workers can write into their index slot.
    results: List[Optional[Dict[str, Any]]] = [None] * len(pairs)
    n_correct_4way = 0
    n_correct_binary = 0
    n_parsed = 0
    n_errors = 0
    n_completed = 0

    # Lock guards mutations of the shared counters + results list + tqdm
    # postfix + incremental write.
    lock = threading.Lock()

    pbar = tqdm(
        total=len(pairs),
        desc=bare_model,
        unit='pair',
        dynamic_ncols=True,
        mininterval=1.0,
    )

    def _on_result(idx: int, result: Dict[str, Any]) -> None:
        nonlocal n_correct_4way, n_correct_binary, n_parsed, n_errors, n_completed
        with lock:
            results[idx] = result
            n_completed += 1
            if result['parsed_answer'] is not None:
                n_parsed += 1
                if result['parsed_answer'] == result['finegrained_label']:
                    n_correct_4way += 1
                if result['parsed_binary'] == result['binary_label']:
                    n_correct_binary += 1
            if result['error']:
                n_errors += 1
                tqdm.write(f"  [{idx+1}/{len(pairs)}] ERROR: {result['error']}")

            acc_4 = n_correct_4way / max(1, n_parsed)
            acc_b = n_correct_binary / max(1, n_parsed)
            pbar.set_postfix({
                'parsed': f'{n_parsed}/{n_completed}',
                '4way': f'{acc_4:.2f}',
                'bin': f'{acc_b:.2f}',
                'err': n_errors,
                'img': result['n_images'],
                'art': result['n_artifacts'],
            })
            pbar.update(1)

            # Incremental write every `progress_every` pairs.
            if n_completed % progress_every == 0 or n_completed == len(pairs):
                non_null = [r for r in results if r is not None]
                _write_output(
                    out_path, model, effective_max_tokens, attach_images,
                    reasoning_kwargs, non_null, n_parsed,
                    n_correct_4way, n_correct_binary, n_completed,
                    temperature=temperature,
                )

    # Worker: calls _process_pair and reports via _on_result.
    def _worker(idx: int, pair: SimilarityPair) -> None:
        try:
            result = _process_pair(
                pair,
                model=model,
                effective_max_tokens=effective_max_tokens,
                temperature=temperature,
                reasoning_kwargs=reasoning_kwargs,
                attach_images=attach_images,
                paper_file_refs=paper_file_refs,
                hash_to_bytes=hash_to_bytes,
                artifact_max_file_bytes=artifact_max_file_bytes,
                artifact_max_total_bytes=artifact_max_total_bytes,
            )
        except Exception as e:
            # Defensive — _process_pair should catch its own errors, but
            # if something escapes, record it so the pair isn't silently
            # dropped.
            result = {
                'eval_pair_id': pair.eval_pair_id,
                'paper_id': pair.paper_id,
                'binary_label': pair.binary_label,
                'finegrained_label': pair.finegrained_label,
                'pair_type': pair.pair_type,
                'parsed_answer': None,
                'parsed_binary': None,
                'response': '',
                'reasoning_content': None,
                'image_paths': [],
                'n_images': 0,
                'artifact_paths': [],
                'n_artifacts': 0,
                'artifact_total_bytes': 0,
                'elapsed_seconds': 0.0,
                'error': f'{type(e).__name__}: {e}',
            }
        _on_result(idx, result)

    # Fan out to the thread pool.
    if concurrency == 1:
        # Sequential path — same thread, no executor overhead. Matches
        # legacy behavior exactly when --concurrency 1.
        for idx, pair in enumerate(pairs):
            _worker(idx, pair)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(_worker, idx, pair)
                for idx, pair in enumerate(pairs)
            ]
            for _ in as_completed(futures):
                pass  # metrics / writes happen in _on_result

    pbar.close()
    acc_4 = n_correct_4way / max(1, n_parsed)
    acc_b = n_correct_binary / max(1, n_parsed)
    print(
        f'\nFinished. Parsed: {n_parsed}/{len(pairs)}  '
        f'4-way accuracy: {acc_4:.3f}  binary accuracy: {acc_b:.3f}  '
        f'errors: {n_errors}'
    )
    non_null = [r for r in results if r is not None]
    _write_output(
        out_path, model, effective_max_tokens, attach_images,
        reasoning_kwargs, non_null, n_parsed,
        n_correct_4way, n_correct_binary, len(pairs),
        temperature=temperature,
    )
    print(f'Wrote {out_path}')


def _write_output(out_path, model, max_tokens, attach_images, reasoning_kwargs,
                  results, n_parsed, n_correct_4way, n_correct_binary,
                  n_completed, temperature=None):
    out_path.write_text(json.dumps({
        'metadata': {
            'mode': '4way',
            'model': model,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'attach_images': attach_images,
            'reasoning_kwargs': reasoning_kwargs,
            'n_pairs': len(results),
            'n_completed': n_completed,
            'n_parsed': n_parsed,
            'n_correct_4way_among_parsed': n_correct_4way,
            'n_correct_binary_among_parsed': n_correct_binary,
            'parse_rate': n_parsed / max(1, n_completed),
            'accuracy_4way': n_correct_4way / max(1, n_parsed),
            'accuracy_binary': n_correct_binary / max(1, n_parsed),
        },
        'results': results,
    }, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description='4-way LLM-as-judge similarity baseline for similarity_check'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='LiteLLM model id, with or without the litellm_proxy/ prefix. '
             'E.g. litellm_proxy/azure_ai/gpt-5.4, '
             'litellm_proxy/gemini/gemini-3.1-pro-preview, '
             'litellm_proxy/anthropic/claude-opus-4-6.',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
                / 'outputs' / 'similarity_check',
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Score only the first N pairs (smoke tests).')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help='Override the completion max_tokens. Default: '
                             'the model\'s catalog-reported max_output_tokens '
                             '(meta_review/model_config.py).')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature. Default 1.0 — required '
                             'by Anthropic extended thinking, recommended '
                             'for Gemini 3 reasoning. Automatically retried '
                             'without temperature if LiteLLM rejects it.')
    parser.add_argument('--no-images', action='store_true',
                        help='Disable image + artifact attachment even for '
                             'multimodal models (text-only baseline).')
    parser.add_argument('--artifact-max-file-bytes', type=int,
                        default=_ARTIFACT_MAX_FILE_BYTES,
                        help='Per-file byte cap for attached text artifacts. '
                             f'Default {_ARTIFACT_MAX_FILE_BYTES}.')
    parser.add_argument('--artifact-max-total-bytes', type=int,
                        default=_ARTIFACT_MAX_TOTAL_BYTES,
                        help='Total byte cap across all attached text '
                             f'artifacts per pair. Default {_ARTIFACT_MAX_TOTAL_BYTES}.')
    parser.add_argument('--concurrency', type=int, default=16,
                        help='Number of concurrent LLM requests in flight. '
                             'Default 16. ThreadPoolExecutor-based; each '
                             'worker has its own retry-with-backoff on 429 '
                             'errors, so higher values are safe to try '
                             '(32, 64, 128...) but may spend time in backoff '
                             'if they trip the provider\'s TPM cap. '
                             'Use 1 for strictly sequential.')
    parser.add_argument('--progress-every', type=int, default=5,
                        help='Print progress every N pairs.')
    args = parser.parse_args()

    run(
        model=args.model,
        output_dir=args.output_dir,
        limit=args.limit,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        no_images=args.no_images,
        artifact_max_file_bytes=args.artifact_max_file_bytes,
        artifact_max_total_bytes=args.artifact_max_total_bytes,
        progress_every=args.progress_every,
        concurrency=args.concurrency,
    )


if __name__ == '__main__':
    main()
