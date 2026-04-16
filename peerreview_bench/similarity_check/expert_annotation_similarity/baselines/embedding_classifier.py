"""
Embedding-based similarity baseline for the similarity_check eval set.

For each pair in the eval set, embed both review item texts using one of
the existing pluggable backends in `embeddings.py` (qwen3, azure, gemini),
compute cosine similarity, and save raw scores. Use `evaluate.py` to turn
these scores into accuracy / AUROC / per-(4cat) breakdowns.

Usage:
    python embedding_classifier.py --backend qwen3
    python embedding_classifier.py --backend azure
    python embedding_classifier.py --backend gemini
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

# Make sibling modules importable
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from load_eval_set import load_similarity_eval_set  # noqa: E402
from embeddings import get_backend  # noqa: E402


def run(backend_name: str, output_dir: Path, limit: int = None) -> None:
    pairs = load_similarity_eval_set()
    print(f'Loaded {len(pairs)} pairs')
    if limit:
        pairs = pairs[:limit]
        print(f'  limited to first {limit}')

    # Collect distinct item texts (item_a/item_b can repeat across pairs)
    text_index = {}
    all_texts = []
    for p in pairs:
        for item in (p.item_a, p.item_b):
            key = (p.paper_id, item.reviewer_id, item.item_number)
            if key not in text_index:
                text_index[key] = len(all_texts)
                all_texts.append(item.text)
    print(f'  {len(all_texts)} unique item texts to embed')

    print(f'Loading backend: {backend_name}')
    backend = get_backend(backend_name)
    print(f'Embedding...')
    emb = backend.embed(all_texts)  # [N, D] L2-normalized
    print(f'  embedding shape: {emb.shape}')

    print('Computing pair cosine similarities...')
    results = []
    for p in pairs:
        idx_a = text_index[(p.paper_id, p.item_a.reviewer_id, p.item_a.item_number)]
        idx_b = text_index[(p.paper_id, p.item_b.reviewer_id, p.item_b.item_number)]
        score = float(np.dot(emb[idx_a], emb[idx_b]))
        results.append({
            'eval_pair_id': p.eval_pair_id,
            'paper_id': p.paper_id,
            'binary_label': p.binary_label,
            'finegrained_label': p.finegrained_label,
            'pair_type': p.pair_type,
            'cosine_score': score,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = backend.name.replace('/', '__').replace(':', '__')
    out_path = output_dir / f'embedding_{safe_name}.json'
    out_path.write_text(json.dumps({
        'metadata': {
            'backend': backend_name,
            'backend_full_name': backend.name,
            'n_pairs': len(results),
            'n_unique_items': len(all_texts),
        },
        'results': results,
    }, indent=2))
    print(f'\nWrote {out_path}')

    sims = [r['cosine_score'] for r in results if r['binary_label'] == 'similar']
    diffs = [r['cosine_score'] for r in results if r['binary_label'] == 'not_similar']
    print(f'\nMean cosine on similar:     {np.mean(sims):.3f} '
          f'(std {np.std(sims):.3f})')
    print(f'Mean cosine on not_similar: {np.mean(diffs):.3f} '
          f'(std {np.std(diffs):.3f})')
    print(f'Gap: {np.mean(sims) - np.mean(diffs):.3f}')


def main():
    parser = argparse.ArgumentParser(
        description='Embedding-based similarity baseline for similarity_check')
    parser.add_argument(
        '--backend', type=str, required=True,
        help='Embedding backend: a short registry name (qwen3, azure, '
             'gemini) OR any LiteLLM embedding model id, with or without '
             'the litellm_proxy/ prefix '
             '(e.g. litellm_proxy/azure_ai/text-embedding-3-large).',
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path(__file__).resolve().parent.parent.parent.parent
                / 'outputs' / 'similarity_check',
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Score only the first N pairs (smoke tests).')
    args = parser.parse_args()
    run(args.backend, args.output_dir, limit=args.limit)


if __name__ == '__main__':
    main()
