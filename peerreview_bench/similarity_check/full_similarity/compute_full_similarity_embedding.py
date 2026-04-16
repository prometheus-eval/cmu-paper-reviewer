"""
Full-similarity compute (EMBEDDING path): within each paper, cosine
similarity between every pair of review items in the expert_annotation HF
config.

Main research question:
    "How similar are AI reviews to human reviews?"

To answer that, we need three pair types (all computed within the same paper):
    - H-H : Human ↔ Human   (inter-human agreement)
    - A-A : AI    ↔ AI      (inter-AI agreement)
    - H-A : Human ↔ AI      (the quantity we actually want)

The output is one JSONL record per pair, carrying:
    {paper_id, pair_type, same_reviewer,
     item_a: {reviewer_id, reviewer_type, review_item_number},
     item_b: {reviewer_id, reviewer_type, review_item_number},
     cosine_score}

plus an `items.json` sidecar that stores each item's review_item text once
so `analyze_embedding.py` can join back to the text if needed.

We reuse the pluggable embedding backends from
`../expert_annotation_similarity/embeddings.py`, so the same Azure /
Gemini / Qwen3 backends work identically across both similarity_check
subdirectories.

A sister script `compute_full_similarity_llm.py` runs the same pair set
through a thinking-mode LLM judge instead of cosine on embeddings — see
its docstring for when to prefer which.

Usage:
    python compute_full_similarity_embedding.py --backend litellm_proxy/azure_ai/text-embedding-3-large
    python compute_full_similarity_embedding.py --backend litellm_proxy/gemini/gemini-embedding-001
    python compute_full_similarity_embedding.py --backend qwen3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent                    # .../similarity_check/full_similarity
_SIMCHECK_DIR = _HERE.parent                                # .../similarity_check
_BENCH_DIR = _SIMCHECK_DIR.parent                           # .../peerreview_bench
_EA_DIR = _SIMCHECK_DIR / 'expert_annotation_similarity'    # reuse embeddings.py

for p in (_HERE, _EA_DIR, _BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from embeddings import get_backend  # noqa: E402
from load_data import load_expert_annotation_rows  # noqa: E402


# ---------------------------------------------------------------------------
# Item prep
# ---------------------------------------------------------------------------

def _dedupe_items(rows: List[Dict]) -> List[Dict]:
    """Dedupe expert_annotation rows by (paper_id, reviewer_id,
    review_item_number). The primary/secondary annotator split duplicates
    each item on the 27 overlap papers; the review TEXT is identical in both
    copies (only the labels differ) so keeping the first occurrence is
    sufficient."""
    seen: Dict[Tuple[int, str, int], Dict] = {}
    for r in rows:
        key = (int(r['paper_id']), r['reviewer_id'], int(r['review_item_number']))
        if key not in seen:
            seen[key] = r
    return list(seen.values())


def _group_by_paper(items: List[Dict]) -> Dict[int, Dict[str, List[int]]]:
    """Return {paper_id: {'H': [idx, ...], 'A': [idx, ...]}} where idx is the
    item's position in the flat `items` list (and therefore its row in the
    embedding matrix)."""
    groups: Dict[int, Dict[str, List[int]]] = defaultdict(lambda: {'H': [], 'A': []})
    for idx, item in enumerate(items):
        pid = int(item['paper_id'])
        tp = 'A' if item['reviewer_type'] == 'AI' else 'H'
        groups[pid][tp].append(idx)
    return dict(groups)


def _item_key_dict(item: Dict) -> Dict:
    """Minimal per-item identity block written into each pair record."""
    return {
        'reviewer_id': item['reviewer_id'],
        'reviewer_type': item['reviewer_type'],
        'review_item_number': int(item['review_item_number']),
    }


# ---------------------------------------------------------------------------
# Main compute
# ---------------------------------------------------------------------------

def run(backend_name: str, output_dir: Path, limit: int = None) -> None:
    print('Loading expert_annotation rows from HuggingFace...')
    rows = load_expert_annotation_rows()
    print(f'  {len(rows)} raw rows (includes primary + secondary duplicates)')

    items = _dedupe_items(rows)
    items = [r for r in items if (r.get('review_item') or '').strip()]
    print(f'  {len(items)} unique non-empty items after dedupe')

    if limit is not None:
        # Smoke-test mode: keep only the first N papers (ordered by paper_id)
        all_pids = sorted({int(r['paper_id']) for r in items})
        keep = set(all_pids[:limit])
        items = [r for r in items if int(r['paper_id']) in keep]
        print(f'  --limit {limit} papers: {len(items)} items retained')

    groups = _group_by_paper(items)
    n_papers = len(groups)
    h_count = sum(len(g['H']) for g in groups.values())
    a_count = sum(len(g['A']) for g in groups.values())
    print(f'  {n_papers} papers, {h_count} Human items, {a_count} AI items')

    # Prepare output paths
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'\nLoading embedding backend: {backend_name}')
    backend = get_backend(backend_name)
    print(f'  backend.name: {backend.name}')

    safe_name = backend.name.replace('/', '__').replace(':', '__')
    pairs_path = output_dir / f'pairs_embedding_{safe_name}.jsonl'
    meta_path = output_dir / f'metadata_embedding_{safe_name}.json'
    items_path = output_dir / 'items.json'  # shared across backends + LLM path

    # Persist the item index once (cheap; ~5 MB). Subsequent backends
    # overwrite with the same content, which is fine.
    items_payload = [
        {
            'paper_id': int(r['paper_id']),
            'reviewer_id': r['reviewer_id'],
            'reviewer_type': r['reviewer_type'],
            'review_item_number': int(r['review_item_number']),
            'review_item': r['review_item'],
        }
        for r in items
    ]
    items_path.write_text(json.dumps(items_payload, indent=2))
    print(f'\nWrote item index: {items_path} ({len(items_payload)} items)')

    texts = [r['review_item'] for r in items]
    t0 = time.time()
    print(f'\nEmbedding {len(texts)} items...')
    emb = backend.embed(texts)  # [N, D], L2-normalized
    embed_seconds = time.time() - t0
    print(f'  embedding shape: {emb.shape}   ({embed_seconds:.1f}s)')

    # Pair generation
    print(f'\nWriting pairs → {pairs_path}')
    n_hh = n_aa = n_ha = 0
    t1 = time.time()
    with pairs_path.open('w') as f:
        for pid in sorted(groups.keys()):
            g = groups[pid]
            H = g['H']
            A = g['A']

            # H × H: upper triangle, skip self-pairs
            for i in range(len(H)):
                for j in range(i + 1, len(H)):
                    ia, ib = H[i], H[j]
                    score = float(np.dot(emb[ia], emb[ib]))
                    rec = {
                        'paper_id': pid,
                        'pair_type': 'H-H',
                        'same_reviewer': items[ia]['reviewer_id'] == items[ib]['reviewer_id'],
                        'item_a': _item_key_dict(items[ia]),
                        'item_b': _item_key_dict(items[ib]),
                        'cosine_score': score,
                    }
                    f.write(json.dumps(rec) + '\n')
                    n_hh += 1

            # A × A: upper triangle, skip self-pairs
            for i in range(len(A)):
                for j in range(i + 1, len(A)):
                    ia, ib = A[i], A[j]
                    score = float(np.dot(emb[ia], emb[ib]))
                    rec = {
                        'paper_id': pid,
                        'pair_type': 'A-A',
                        'same_reviewer': items[ia]['reviewer_id'] == items[ib]['reviewer_id'],
                        'item_a': _item_key_dict(items[ia]),
                        'item_b': _item_key_dict(items[ib]),
                        'cosine_score': score,
                    }
                    f.write(json.dumps(rec) + '\n')
                    n_aa += 1

            # H × A: full rectangular grid (every H × every A on this paper)
            for i in H:
                for j in A:
                    score = float(np.dot(emb[i], emb[j]))
                    rec = {
                        'paper_id': pid,
                        'pair_type': 'H-A',
                        'same_reviewer': False,  # always different by construction
                        'item_a': _item_key_dict(items[i]),
                        'item_b': _item_key_dict(items[j]),
                        'cosine_score': score,
                    }
                    f.write(json.dumps(rec) + '\n')
                    n_ha += 1

    pair_seconds = time.time() - t1
    total = n_hh + n_aa + n_ha
    print(f'  H-H pairs : {n_hh}')
    print(f'  A-A pairs : {n_aa}')
    print(f'  H-A pairs : {n_ha}')
    print(f'  TOTAL     : {total}')
    print(f'  ({pair_seconds:.1f}s to score + write)')

    # Metadata
    meta = {
        'backend': backend_name,
        'backend_full_name': backend.name,
        'n_papers': n_papers,
        'n_items': len(items),
        'n_human_items': h_count,
        'n_ai_items': a_count,
        'n_pairs': total,
        'n_hh_pairs': n_hh,
        'n_aa_pairs': n_aa,
        'n_ha_pairs': n_ha,
        'embed_seconds': round(embed_seconds, 1),
        'pair_write_seconds': round(pair_seconds, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f'\nWrote metadata: {meta_path}')

    # Quick sanity summary so users can eyeball if the numbers are plausible
    # before running analyze.py.
    def _mean(xs):
        return float(np.mean(xs)) if xs else float('nan')

    hh_scores, aa_scores, ha_scores = [], [], []
    with pairs_path.open() as f:
        for line in f:
            rec = json.loads(line)
            s = rec['cosine_score']
            pt = rec['pair_type']
            if pt == 'H-H':
                hh_scores.append(s)
            elif pt == 'A-A':
                aa_scores.append(s)
            else:
                ha_scores.append(s)

    print('\n=== Quick sanity summary (mean cosine by pair type) ===')
    print(f'  H-H: {_mean(hh_scores):+.3f}')
    print(f'  A-A: {_mean(aa_scores):+.3f}')
    print(f'  H-A: {_mean(ha_scores):+.3f}')
    print('(Run analyze.py for the full per-paper / per-model breakdown.)')


def main():
    parser = argparse.ArgumentParser(
        description='Compute full within-paper pairwise similarity on the '
                    'expert_annotation HF config.')
    parser.add_argument(
        '--backend', type=str, required=True,
        help='Embedding backend: short name (qwen3, azure, gemini) OR any '
             'LiteLLM embedding model id (with or without the `litellm_proxy/` prefix).',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=_BENCH_DIR / 'outputs' / 'full_similarity',
        help='Directory to write pairs_*.jsonl, metadata_*.json, items.json',
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Only use the first N papers (smoke test).',
    )
    args = parser.parse_args()
    run(args.backend, args.output_dir, limit=args.limit)


if __name__ == '__main__':
    main()
