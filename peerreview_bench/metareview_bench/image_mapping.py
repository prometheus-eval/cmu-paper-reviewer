"""
Map review-item text to the subset of paper images that a meta-reviewer
would plausibly need.

Approach:
  1. Parse `review_content`, `review_claim`, and `review_evidence` for
     explicit figure / table references (e.g. "Fig. 2", "Figure 3b",
     "Table 5", "Supplementary Figure 1").
  2. Load the paper's images_list.json (present in most preprint/ dirs).
  3. For each reference in the review text, find an images_list entry whose
     `img_path` or `caption` matches.
  4. Return the list of selected `{img_path, caption, bytes}` dicts.

If no explicit references are found, we return ALL images in the paper so
the LLM still gets complete visual context.

By default, `max_images=None` — meaning there is no cap and every matching
(or every fallback) image is returned. Callers who want a tighter budget
for cost/latency reasons can pass a positive int.

Image bytes come from the `submitted_papers` HF config (via a hash->bytes
map), so this module doesn't touch the local filesystem at all.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional


_FIG_REF_RE = re.compile(
    r"""
    (?:Supplementary\s+)?   # optional "Supplementary"
    (?:Fig(?:ure)?|Table)   # Fig / Figure / Table
    s?                      # optional plural 's'
    \.?                     # optional "."
    \s*                     # whitespace
    ([SS]?\d+[A-Za-z]?)     # the number, optionally with an 'S' prefix and suffix letter
    """,
    re.IGNORECASE | re.VERBOSE,
)


def find_figure_references(text: str) -> List[str]:
    """Return the distinct figure/table reference numbers mentioned in `text`."""
    if not text:
        return []
    refs: List[str] = []
    seen = set()
    for m in _FIG_REF_RE.finditer(text):
        num = m.group(1).lower()
        if num not in seen:
            seen.add(num)
            refs.append(num)
    return refs


def _load_images_list_from_refs(
    file_refs: List[Dict[str, Any]],
    hash_to_bytes: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    """Find the `images_list.json` entry in file_refs, decode it, and return
    the parsed list. Returns None if there's no images_list.json for this paper."""
    for ref in file_refs:
        path = ref.get("path", "") if isinstance(ref, dict) else ""
        if path.endswith("images_list.json"):
            blob = hash_to_bytes.get(ref["content_hash"])
            if blob is None:
                return None
            content = blob["content_bytes"] if isinstance(blob, dict) else blob
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                return None
    return None


def _file_ref_by_path(
    file_refs: List[Dict[str, Any]],
    path: str,
) -> Optional[Dict[str, Any]]:
    """Look up a ref by its relative path (exact or suffix match)."""
    path = path.lstrip("./")
    for ref in file_refs:
        r_path = ref.get("path", "") if isinstance(ref, dict) else ""
        if r_path == path or r_path.endswith("/" + path) or path.endswith(r_path):
            return ref
    return None


def select_images_for_review_item(
    review_text: str,
    file_refs: List[Dict[str, Any]],
    hash_to_bytes: Dict[str, Any],
    *,
    max_images: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return relevant image entries for a review item.

    Each returned dict has: {img_path, caption, bytes (bytes|None), mime}.

    If `max_images` is None (the default), there is NO cap: every image
    matched against a figure reference, and — if no references are found —
    every image in the paper, is returned. Pass a positive int to cap.
    """
    images_list = _load_images_list_from_refs(file_refs, hash_to_bytes)
    if not images_list:
        return []

    refs = find_figure_references(review_text)
    selected: List[Dict[str, Any]] = []

    def _mime_for_path(path: str) -> str:
        ext = path.rsplit(".", 1)[-1].lower()
        if ext in ("jpg", "jpeg"):
            return "image/jpeg"
        if ext == "png":
            return "image/png"
        if ext == "gif":
            return "image/gif"
        if ext == "webp":
            return "image/webp"
        return "application/octet-stream"

    def _resolve(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        img_path = entry.get("img_path", "") if isinstance(entry, dict) else ""
        if not img_path:
            return None
        ref = _file_ref_by_path(file_refs, img_path)
        if ref is None:
            return None
        blob = hash_to_bytes.get(ref["content_hash"])
        if blob is None:
            return None
        content = blob["content_bytes"] if isinstance(blob, dict) else blob
        if not isinstance(content, (bytes, bytearray)):
            return None
        return {
            "img_path": img_path,
            "caption": entry.get("caption", ""),
            "bytes": bytes(content),
            "mime": _mime_for_path(img_path),
        }

    def _room_left() -> bool:
        return max_images is None or len(selected) < max_images

    # If the review text referenced specific figures, select those first.
    if refs:
        for entry in images_list:
            if not _room_left():
                break
            path = entry.get("img_path", "").lower()
            caption = entry.get("caption", "").lower()
            for ref in refs:
                ref_norm = ref.lower()
                if f"figure_{ref_norm}" in path or f"figure {ref_norm}" in caption or \
                   f"fig. {ref_norm}" in caption or f"fig{ref_norm}" in path or \
                   f"table_{ref_norm}" in path or f"table {ref_norm}" in caption:
                    resolved = _resolve(entry)
                    if resolved and resolved not in selected:
                        selected.append(resolved)
                    break

    # Fallback: if nothing matched, include every image in the paper (or
    # the first `max_images` if a cap was explicitly requested) so the
    # model has full visual context.
    if not selected:
        pool = images_list if max_images is None else images_list[:max_images]
        for entry in pool:
            resolved = _resolve(entry)
            if resolved:
                selected.append(resolved)

    if max_images is None:
        return selected
    return selected[:max_images]
