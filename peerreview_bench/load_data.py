"""
Shared loader for the PeerReview Bench HuggingFace dataset.

Top-level entry points (used by analysis/, meta_review/, similarity_check/):

    load_annotations(annotator_source='both')      -> expert_annotation config
    load_meta_reviewer()                            -> meta_reviewer config
    load_reviewer()                                 -> reviewer config
    load_submitted_papers()                         -> submitted_papers config
    load_dataframe(annotator_source='both')         -> expert_annotation as a DataFrame

All loaders read from the `prometheus-eval/peerreview-bench` HuggingFace
dataset. There is no longer a local-JSON fallback; if HuggingFace is
unreachable the loaders raise a clear RuntimeError pointing at the upload
command.

By default, `load_annotations()` loads both primary and secondary
annotations as independent rows ("merged" semantics): items from the 27
overlap papers contribute two data points (one per annotator), and items
from the other papers contribute one. Pass `annotator_source='primary'` or
`'secondary'` to load only one side.

The per-paper Best/Worst Human rankings live in the local file
`reviewer_rankings.json` (next to this module). Those were dropped from
the HF `expert_annotation` schema to keep the schema clean; the loader
rehydrates them from the local file. Keys in that file are already the
HF paper_ids (1..85), so no translation is needed.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# HuggingFace Hub Xet workaround
# ---------------------------------------------------------------------------
# The new default Xet transfer backend (huggingface_hub >= 0.30, fully
# enforced in >= 1.0) has known hangs on large multi-shard datasets — in
# particular, `load_dataset` and `push_to_hub` can stall indefinitely at
# 0% or near 99% when a shard transfer stalls (refs:
# https://github.com/huggingface/xet-core/issues/527
# https://github.com/huggingface/datasets/issues/5990
# https://discuss.huggingface.co/t/huggingface-dataset-download-stuck-in-kaggle/175183).
#
# The workaround is to disable Xet and fall back to the plain HTTP path.
# These env vars MUST be set before `huggingface_hub` is imported — it
# snapshots them at import time — so we set them here at module-import time
# (not inside the loader functions) unless the caller has explicitly opted
# in by exporting the var themselves.
#
#   HF_HUB_DISABLE_XET=1         -> disables Xet routing (the key fix)
#   HF_HUB_ENABLE_HF_TRANSFER=0  -> kept for compatibility with <1.0; a
#                                   no-op on >=1.0 (the hf_transfer path
#                                   was replaced by HF_XET_HIGH_PERFORMANCE)
#   HF_HUB_DOWNLOAD_TIMEOUT=120  -> bump the too-short 10s default
#
# Users can opt out of any of these by exporting the var with a different
# value before running the script.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

DATASET_REPO = 'prometheus-eval/peerreview-bench'
CONFIG_NAME = 'expert_annotation'


# ---------------------------------------------------------------------------
# Schema stale-cache guard: HF_FORCE_REDOWNLOAD
# ---------------------------------------------------------------------------
# When the dataset's schema changes on the Hub (e.g., a column is renamed),
# users with a stale local cache hit KeyErrors because `load_dataset` returns
# the previously cached rows with the old column names. We guard against this
# by defaulting to `download_mode=FORCE_REDOWNLOAD` on every load_dataset call
# in this module. That forces a fresh pull from the Hub on every process
# start, which catches any upstream schema change.
#
# For fast iteration where the schema is known to be stable locally, users
# can opt out by exporting `HF_FORCE_REDOWNLOAD=0`. The large
# `submitted_papers` config (~2 GB of blob storage) is intentionally
# EXEMPT from this guard — its schema is stable and re-downloading it on
# every run would waste bandwidth. See `load_submitted_papers` below.

def _download_mode():
    """Return the datasets DownloadMode to use for schema-bearing configs.

    Default: FORCE_REDOWNLOAD (protects against stale caches after upstream
    schema changes). Set HF_FORCE_REDOWNLOAD=0 to opt out for fast local
    iteration.
    """
    from datasets import DownloadMode
    if os.environ.get('HF_FORCE_REDOWNLOAD', '1') == '0':
        return DownloadMode.REUSE_DATASET_IF_EXISTS
    return DownloadMode.FORCE_REDOWNLOAD

AnnotatorSource = Literal['primary', 'secondary', 'both']

# `reviewer_rankings.json` is the only local sidecar we still read — it
# carries per-paper Best/Worst Human tags that were dropped from the HF
# `expert_annotation` schema. Lives alongside this module.
_HERE = Path(__file__).resolve().parent
LOCAL_RANKINGS = _HERE / 'reviewer_rankings.json'


@dataclass
class ReviewItem:
    """Canonical item schema used by all analysis scripts."""
    paper_id: int
    item_id: str
    reviewer_id: str
    reviewer_type: str
    model_name: Optional[str]
    item_number: int
    correctness: Optional[str]
    significance: Optional[str]
    evidence: Optional[str]
    correctness_numeric: Optional[int]
    significance_numeric: Optional[int]
    evidence_numeric: Optional[int]


# ============================================================================
# LABEL → NUMERIC CONVERSION
# ============================================================================
#
# The published HF schema carries string labels only (`correctness`,
# `significance`, `evidence`). The analysis scripts work on a canonical
# ReviewItem dataclass that still exposes `correctness_numeric`,
# `significance_numeric`, `evidence_numeric` for numeric comparisons, so we
# compute the numeric values from the strings at load time.

_CORR_TO_NUM = {'Correct': 1, 'Not Correct': 0}
_SIG_TO_NUM = {
    'Very Significant': 2,   # legacy label, merged into 'Significant' at upload
    'Significant': 2,
    'Marginally Significant': 1,
    'Not Significant': 0,
}
_EVI_TO_NUM = {'Sufficient': 1, 'Requires More': 0}


def _corr_num(label: Optional[str]) -> Optional[int]:
    return _CORR_TO_NUM.get(label) if label else None


def _sig_num(label: Optional[str]) -> Optional[int]:
    return _SIG_TO_NUM.get(label) if label else None


def _evi_num(label: Optional[str]) -> Optional[int]:
    return _EVI_TO_NUM.get(label) if label else None


# ============================================================================
# HF LOADER
# ============================================================================

def _load_from_hf(annotator_source: str) -> Tuple[List[ReviewItem], Dict[int, Dict[str, str]]]:
    """Load from the HuggingFace dataset. Raises on failure (caller decides fallback).

    If annotator_source == 'both', include every row. Otherwise, filter by
    the specified annotator_source.

    Note on field mapping: the published HF schema renamed `item_number` to
    `review_item_number` and dropped the `*_numeric`, `reviewer_rank`, and
    `validity_category` columns. We remap to the ReviewItem dataclass here so
    downstream analysis code keeps working unchanged — numeric fields are
    recomputed from the string labels, and rankings are loaded from the local
    `reviewer_rankings.json` instead of from the row's (now-gone) column.
    """
    from datasets import load_dataset
    ds = load_dataset(DATASET_REPO, CONFIG_NAME, split='eval',
                      download_mode=_download_mode())

    items: List[ReviewItem] = []

    for row in ds:
        if annotator_source != 'both' and row.get('annotator_source') != annotator_source:
            continue

        pid = int(row['paper_id'])
        src = row.get('annotator_source', '')
        item_num = int(row.get('review_item_number', row.get('item_number', 0)))
        corr_label = row.get('correctness')
        sig_label = row.get('significance')
        evi_label = row.get('evidence')
        items.append(ReviewItem(
            paper_id=pid,
            item_id=f"paper{pid}_{row['reviewer_id']}_item{item_num}_{src}",
            reviewer_id=row['reviewer_id'],
            reviewer_type=row['reviewer_type'],
            model_name=row['reviewer_id'] if row['reviewer_type'] == 'AI' else None,
            item_number=item_num,
            correctness=corr_label,
            significance=sig_label,
            evidence=evi_label,
            correctness_numeric=_corr_num(corr_label),
            significance_numeric=_sig_num(sig_label),
            evidence_numeric=_evi_num(evi_label),
        ))

    # Reviewer rankings are no longer in the expert_annotation schema —
    # load them from the local `reviewer_rankings.json`. Keys in that file
    # are the HF paper_ids (1..85); no translation needed.
    rankings: Dict[int, Dict[str, str]] = {}
    if LOCAL_RANKINGS.exists():
        with open(LOCAL_RANKINGS) as f:
            raw = json.load(f)
        for k, v in raw.items():
            if k.startswith('_'):
                continue
            try:
                pid = int(k)
            except ValueError:
                continue
            if isinstance(v, dict) and 'best' in v:
                rankings[pid] = {'best': v['best'], 'worst': v.get('worst')}

    return items, rankings


# ============================================================================
# PUBLIC API
# ============================================================================

def _maybe_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def load_annotations(annotator_source: str = 'both') -> Tuple[List[ReviewItem], Dict[int, Dict[str, str]]]:
    """Load items and rankings from the HuggingFace dataset.

    Args:
        annotator_source: 'primary', 'secondary', or 'both' (default).
            'both' includes every annotation as an independent data point,
            so items from the 27 overlap papers contribute twice (once per
            annotator) while items from the other papers contribute once.
            This is the "merged primary + secondary" semantics used by
            the main analysis scripts.

    Returns:
        (items, rankings) where items is a list of ReviewItem and rankings is
        {paper_id: {'best': reviewer_id, 'worst': reviewer_id}}.

    HuggingFace-only. On failure, raises a RuntimeError telling the caller
    how to push the dataset first. There is no local-JSON fallback.
    """
    if annotator_source not in ('primary', 'secondary', 'both'):
        raise ValueError(
            f"annotator_source must be 'primary', 'secondary', or 'both' "
            f"(got {annotator_source!r})"
        )
    try:
        return _load_from_hf(annotator_source)
    except Exception as e:
        raise RuntimeError(
            f"Could not load `expert_annotation` config from {DATASET_REPO} "
            f"({type(e).__name__}: {e}).\n\n"
            f"The analysis pipeline is HuggingFace-only. The dataset must "
            f"have been pushed upstream before running any analysis script."
        ) from e


def load_dataframe(annotator_source: str = 'both'):
    """Load as a pandas DataFrame (used by the GLMM script).

    Same `annotator_source` semantics as `load_annotations`. Defaults to
    'both', meaning both primary and secondary rows are included.
    """
    import pandas as pd
    items, rankings = load_annotations(annotator_source)
    rows = [{
        'paper_id': i.paper_id,
        'item_id': i.item_id,
        'reviewer_id': i.reviewer_id,
        'reviewer_type': i.reviewer_type,
        'model_name': i.model_name,
        'item_number': i.item_number,
        'correctness': i.correctness_numeric,
        'significance': i.significance_numeric,
        'evidence': i.evidence_numeric,
    } for i in items]
    return pd.DataFrame(rows), rankings


# ============================================================================
# HUGGINGFACE LOADERS FOR OTHER CONFIGS
#
# These always load from HuggingFace (no local fallback). If you're running
# offline before the upload has happened, you'll need to either push the
# dataset first or feed the raw annotation JSONs into your own preprocessing.
# ============================================================================

def load_expert_annotation_rows() -> List[Dict[str, Any]]:
    """Load the `expert_annotation` config as a list of raw HF row dicts.

    This returns the FULL schema including `review_content`, `review_claim`,
    `review_evidence`, etc. — unlike `load_annotations()` which returns
    `ReviewItem` dataclasses without the review text.

    Always loads from HuggingFace (no local fallback).
    """
    from datasets import load_dataset
    ds = load_dataset(DATASET_REPO, CONFIG_NAME, split='eval',
                      download_mode=_download_mode())
    return [dict(row) for row in ds]


def load_meta_reviewer() -> List[Dict[str, Any]]:
    """Load the `meta_reviewer` config as a list of dicts.

    Each row corresponds to one review item from the 27 overlap papers, with:
      - paper_id, paper_title, paper_content, file_refs
      - reviewer_id, reviewer_type, item_number
      - review_content, review_claim, review_evidence, review_cited_references
      - per-annotator labels (correctness_primary, correctness_secondary, ...)
      - collapsed 10-class label (`label`, `label_id`)

    HuggingFace-only. If the dataset isn't pushed yet, this raises a clear
    RuntimeError. There is no local-filesystem fallback.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_REPO, 'meta_reviewer', split='eval',
                          download_mode=_download_mode())
        return [dict(row) for row in ds]
    except Exception as e:
        raise RuntimeError(
            f"Could not load `meta_reviewer` config from {DATASET_REPO} "
            f"({type(e).__name__}: {e}).\n\n"
            f"The meta_review pipeline is HuggingFace-only — there is no "
            f"local fallback. The dataset must have been pushed upstream "
            f"before running any analysis script."
        ) from e


def load_reviewer() -> List[Dict[str, Any]]:
    """Load the `reviewer` config (one row per paper, minimal schema)."""
    from datasets import load_dataset
    ds = load_dataset(DATASET_REPO, 'reviewer', split='eval',
                      download_mode=_download_mode())
    return [dict(row) for row in ds]


def load_submitted_papers() -> Dict[str, Dict[str, Any]]:
    """Load the `submitted_papers` config as a hash -> file dict.

    Returns:
        {content_hash: {'content_bytes': bytes, 'size_bytes': int, 'is_text': bool}}

    Useful for resolving `file_refs` from the other configs.

    NOTE: this loader is intentionally EXEMPT from the HF_FORCE_REDOWNLOAD
    guard. `submitted_papers` is ~2 GB of blob storage with a stable schema
    (`{content_hash, content_bytes, size_bytes, is_text}`) — re-downloading
    it on every run would waste bandwidth. If the schema ever changes, you
    can force a refresh by wiping the cache manually or by passing
    `download_mode=DownloadMode.FORCE_REDOWNLOAD` here.
    """
    from datasets import load_dataset
    ds = load_dataset(DATASET_REPO, 'submitted_papers', split='eval')
    out: Dict[str, Dict[str, Any]] = {}
    for row in ds:
        out[row['content_hash']] = {
            'content_bytes': row['content_bytes'],
            'size_bytes': int(row['size_bytes']),
            'is_text': bool(row['is_text']),
        }
    return out


def resolve_file_refs(file_refs: List[Dict[str, Any]],
                      hash_to_file: Dict[str, Dict[str, Any]]) -> Dict[str, bytes]:
    """Given a list of file_ref dicts and a hash->file index from
    load_submitted_papers(), return {path: content_bytes} for the referenced files."""
    out: Dict[str, bytes] = {}
    for ref in file_refs:
        h = ref.get('content_hash') if isinstance(ref, dict) else None
        if h and h in hash_to_file:
            out[ref['path']] = hash_to_file[h]['content_bytes']
    return out
