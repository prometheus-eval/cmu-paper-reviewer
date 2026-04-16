"""
Loader for the `similarity_check` subset of PeerReview Bench.

Loads from the HuggingFace dataset `prometheus-eval/peerreview-bench`,
config `similarity_check`, split `eval`. Returns one dataclass per pair
with both the binary label (matching the annotator's implicit judgment)
and the fine-grained label from the audit.

Schema columns (see upload_to_hf.py for full docs):
  eval_pair_id, source_pair_id, paper_id, paper_title, paper_content,
  item_a_*, item_b_*, binary_label, finegrained_label,
  pair_type, rationale, source_bucket

`file_refs` is NOT inlined in the similarity_check rows. To fetch
supplementary code/data files, join on `paper_id` against the `reviewer`
config — see `load_paper_file_refs()` below.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Same Xet workaround as load_data.py — see that file for the rationale.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

DATASET_REPO = 'prometheus-eval/peerreview-bench'
CONFIG_NAME = 'similarity_check'


# ---------------------------------------------------------------------------
# Schema stale-cache guard — see load_data.py for the full rationale.
# ---------------------------------------------------------------------------

def _download_mode():
    """Return the datasets DownloadMode for schema-bearing configs.

    Default: FORCE_REDOWNLOAD (protects against stale caches after upstream
    schema changes — e.g. the four_cat_label → finegrained_label rename).
    Set HF_FORCE_REDOWNLOAD=0 to opt out for fast local iteration.
    """
    from datasets import DownloadMode
    if os.environ.get('HF_FORCE_REDOWNLOAD', '1') == '0':
        return DownloadMode.REUSE_DATASET_IF_EXISTS
    return DownloadMode.FORCE_REDOWNLOAD


@dataclass
class ReviewItem:
    reviewer_id: str
    reviewer_type: str
    item_number: int
    text: str


@dataclass
class SimilarityPair:
    eval_pair_id: str
    source_pair_id: str
    paper_id: int
    paper_title: str
    paper_content: str
    item_a: ReviewItem
    item_b: ReviewItem
    binary_label: str            # "similar" | "not_similar"
    finegrained_label: str       # one of FINEGRAINED_LABELS below
    pair_type: str               # "AI-AI" | "AI-Human"
    rationale: str
    source_bucket: str


# Allowed values for finegrained_label (descriptive strings)
FINEGRAINED_LABELS = (
    'same subject, same argument, same evidence',     # near-paraphrase
    'same subject, same argument, different evidence',# convergent conclusion
    'same subject, different argument',                # topical neighbor
    'different subject',                               # unrelated
)

# Mapping from finegrained_label to short audit code (for backwards-compat)
FINEGRAINED_TO_SHORT = {
    'same subject, same argument, same evidence': 'c',
    'same subject, same argument, different evidence': 'b',
    'same subject, different argument': 'a',
    'different subject': 'd',
}


def _row_to_pair(row: Dict[str, Any]) -> SimilarityPair:
    return SimilarityPair(
        eval_pair_id=row['eval_pair_id'],
        source_pair_id=row['source_pair_id'],
        paper_id=int(row['paper_id']),
        paper_title=row['paper_title'],
        paper_content=row['paper_content'],
        item_a=ReviewItem(
            reviewer_id=row['item_a_reviewer_id'],
            reviewer_type=row['item_a_reviewer_type'],
            item_number=int(row['item_a_item_number']),
            text=row['item_a_text'],
        ),
        item_b=ReviewItem(
            reviewer_id=row['item_b_reviewer_id'],
            reviewer_type=row['item_b_reviewer_type'],
            item_number=int(row['item_b_item_number']),
            text=row['item_b_text'],
        ),
        binary_label=row['binary_label'],
        finegrained_label=row['finegrained_label'],
        pair_type=row['pair_type'],
        rationale=row.get('rationale', '') or '',
        source_bucket=row['source_bucket'],
    )


def load_paper_file_refs() -> Dict[int, List[Dict[str, Any]]]:
    """Convenience helper: load the `reviewer` config and return a
    {paper_id: file_refs} index for callers that need supplementary
    code/data files for the similarity_check papers."""
    from datasets import load_dataset
    ds = load_dataset(DATASET_REPO, 'reviewer', split='eval',
                      download_mode=_download_mode())
    return {int(r['paper_id']): r['file_refs'] for r in ds}


def load_similarity_eval_set() -> List[SimilarityPair]:
    """Load the similarity_check evaluation set from HuggingFace.

    Returns the 164 pairs currently in the HF dataset. No local filtering
    is applied — whatever is on the Hub is what you get. The post-hoc
    label-quality filter that removed 74 of the original 238 pairs was
    applied at upload time (see upload_to_hf.py::DROP_PAIR_IDS_SIMILARITY_CHECK)
    so it does not need to be re-applied here.
    """
    from datasets import load_dataset
    try:
        ds = load_dataset(DATASET_REPO, CONFIG_NAME, split='eval',
                          download_mode=_download_mode())
    except Exception as e:
        raise RuntimeError(
            f'Could not load {CONFIG_NAME!r} config from {DATASET_REPO} '
            f'({type(e).__name__}: {e}). Make sure the dataset has been '
            f'pushed by upload_to_hf.py first.'
        ) from e
    return [_row_to_pair(dict(row)) for row in ds]


def load_similarity_eval_set_raw() -> List[Dict[str, Any]]:
    """Same as load_similarity_eval_set but returns raw dicts instead of
    dataclasses. Useful for callers that want to keep the HF row schema
    intact for downstream tooling."""
    from datasets import load_dataset
    ds = load_dataset(DATASET_REPO, CONFIG_NAME, split='eval',
                      download_mode=_download_mode())
    return [dict(row) for row in ds]


if __name__ == '__main__':
    pairs = load_similarity_eval_set()
    print(f'Loaded {len(pairs)} pairs from {DATASET_REPO}/{CONFIG_NAME}')
    from collections import Counter
    print(f'  binary: {dict(Counter(p.binary_label for p in pairs))}')
    print(f'  finegrained: {dict(Counter(p.finegrained_label for p in pairs))}')
    print(f'  pair_type: {dict(Counter(p.pair_type for p in pairs))}')
    print(f'  papers: {len(set(p.paper_id for p in pairs))}')
