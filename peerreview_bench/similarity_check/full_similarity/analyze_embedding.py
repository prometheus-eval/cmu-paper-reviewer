"""
Aggregate the per-pair JSONL produced by compute_full_similarity_embedding.py
into paper-level and dataset-level statistics that answer:

    "How similar is AI review compared to human reviews?"

Input  : pairs_embedding_{backend}.jsonl + items.json
Output : analysis_embedding_{backend}.json — a structured report covering

    1. Global pair-type summary (mean/median/std/n of H-H, A-A, H-A)
    2. Same-reviewer vs different-reviewer split within H-H and A-A
    3. Per-AI-model "human-likeness"       (mean H-A cosine where A is from model X)
    4. Per-AI-model inter-AI agreement     (mean A-A cosine, split by model pair)
    5. Nearest-neighbor analysis
       - For each AI item: max cosine to any Human item on the same paper
       - For each Human item: max cosine to any AI item on the same paper
    6. Coverage at thresholds: P(an AI item has a Human match ≥ τ), τ ∈ {.5, .6, .7, .8}
    7. Per-paper summary                    (mean by pair_type, sortable)
    8. AI vs Human gap test
       Compare the distribution of H-A cosines to H-H (the natural ceiling) —
       do AIs come as close to humans as humans come to each other?

Usage:
    python analyze_embedding.py outputs/full_similarity/pairs_embedding_<backend>.jsonl
    python analyze_embedding.py outputs/full_similarity/pairs_embedding_<backend>.jsonl --out analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _stream_pairs(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _summarize(xs: List[float]) -> Dict:
    if not xs:
        return {'n': 0}
    arr = np.asarray(xs, dtype=np.float32)
    return {
        'n': int(arr.size),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        'p10': float(np.percentile(arr, 10)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


def _rank_by_mean(rows: List[Tuple[str, List[float]]]) -> List[Dict]:
    """Given [(label, [scores]), ...] return one dict per label sorted by mean desc."""
    out = []
    for label, scores in rows:
        if not scores:
            continue
        out.append({'label': label, **_summarize(scores)})
    out.sort(key=lambda r: r.get('mean', float('-inf')), reverse=True)
    return out


def analyze(pairs_path: Path, items_path: Path, out_path: Path) -> None:
    print(f'Loading pairs from {pairs_path}...')

    # ------------------------------------------------------------------
    # Pass 1: collect per-pair scores into bucketed lists
    # ------------------------------------------------------------------
    hh_scores: List[float] = []
    aa_scores: List[float] = []
    ha_scores: List[float] = []

    # H-H split
    hh_same: List[float] = []   # same reviewer (within-reviewer consistency)
    hh_diff: List[float] = []   # different reviewers (inter-human agreement)
    # A-A split
    aa_same: List[float] = []   # same AI model (same-model consistency)
    aa_diff: List[float] = []   # different AI models (inter-model agreement)

    # Per-AI-model indexed by the AI item's reviewer_id
    per_model_ha_scores: Dict[str, List[float]] = defaultdict(list)
    per_model_hh_per_model_gap: Dict[str, List[float]] = defaultdict(list)  # not used yet
    # Per inter-model pair (sorted tuple of models, e.g. ('Claude','GPT'))
    per_pair_aa_scores: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    # Nearest-neighbor tracking: for each (paper_id, AI item key),
    # keep running max over any H partner; same for (paper_id, H item key) over A partners.
    #   key = (paper_id, reviewer_id, review_item_number)
    ai_max_to_human: Dict[Tuple[int, str, int], float] = {}
    human_max_to_ai: Dict[Tuple[int, str, int], float] = {}

    # Per-paper accumulators
    per_paper: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {'H-H': [], 'A-A': [], 'H-A': []}
    )

    n_total = 0
    for rec in _stream_pairs(pairs_path):
        n_total += 1
        pt = rec['pair_type']
        s = rec['cosine_score']
        pid = rec['paper_id']
        same_rev = rec.get('same_reviewer', False)
        a = rec['item_a']
        b = rec['item_b']

        per_paper[pid][pt].append(s)

        if pt == 'H-H':
            hh_scores.append(s)
            (hh_same if same_rev else hh_diff).append(s)
        elif pt == 'A-A':
            aa_scores.append(s)
            (aa_same if same_rev else aa_diff).append(s)
            m1, m2 = sorted([a['reviewer_id'], b['reviewer_id']])
            per_pair_aa_scores[(m1, m2)].append(s)
        elif pt == 'H-A':
            ha_scores.append(s)
            # Identify the AI side robustly: one side is Human, the other is AI.
            if a['reviewer_type'] == 'AI':
                ai_item, hu_item = a, b
            else:
                ai_item, hu_item = b, a
            per_model_ha_scores[ai_item['reviewer_id']].append(s)

            ai_key = (pid, ai_item['reviewer_id'], ai_item['review_item_number'])
            hu_key = (pid, hu_item['reviewer_id'], hu_item['review_item_number'])
            if s > ai_max_to_human.get(ai_key, float('-inf')):
                ai_max_to_human[ai_key] = s
            if s > human_max_to_ai.get(hu_key, float('-inf')):
                human_max_to_ai[hu_key] = s
        else:
            # Unexpected type — skip but note it
            print(f'WARNING: unknown pair_type {pt!r}', file=sys.stderr)

    print(f'  read {n_total} pair records')
    print(f'  H-H = {len(hh_scores)}, A-A = {len(aa_scores)}, H-A = {len(ha_scores)}')

    # ------------------------------------------------------------------
    # 1. Global pair-type summary
    # ------------------------------------------------------------------
    global_summary = {
        'H-H': _summarize(hh_scores),
        'A-A': _summarize(aa_scores),
        'H-A': _summarize(ha_scores),
    }

    # ------------------------------------------------------------------
    # 2. Same-reviewer vs different-reviewer split
    # ------------------------------------------------------------------
    split_summary = {
        'H-H_same_reviewer': _summarize(hh_same),
        'H-H_diff_reviewer': _summarize(hh_diff),
        'A-A_same_model':    _summarize(aa_same),
        'A-A_diff_model':    _summarize(aa_diff),
    }

    # ------------------------------------------------------------------
    # 3. Per-AI-model human-likeness ranking
    # ------------------------------------------------------------------
    per_model_ha = _rank_by_mean(sorted(per_model_ha_scores.items()))

    # ------------------------------------------------------------------
    # 4. Per-AI-model-pair inter-AI agreement
    # ------------------------------------------------------------------
    per_pair_aa = []
    for (m1, m2), scores in sorted(per_pair_aa_scores.items()):
        label = f'{m1} × {m2}' if m1 != m2 else f'{m1} (same-model)'
        per_pair_aa.append({'label': label, 'model_a': m1, 'model_b': m2, **_summarize(scores)})
    per_pair_aa.sort(key=lambda r: r.get('mean', float('-inf')), reverse=True)

    # ------------------------------------------------------------------
    # 5. Nearest-neighbor distribution
    # ------------------------------------------------------------------
    ai_nn_scores = list(ai_max_to_human.values())
    human_nn_scores = list(human_max_to_ai.values())

    nn_summary = {
        'ai_item_max_to_human': _summarize(ai_nn_scores),
        'human_item_max_to_ai': _summarize(human_nn_scores),
    }

    # Nearest-neighbor per AI model
    ai_nn_per_model: Dict[str, List[float]] = defaultdict(list)
    for (pid, rid, itn), s in ai_max_to_human.items():
        ai_nn_per_model[rid].append(s)
    nn_per_model = _rank_by_mean(sorted(ai_nn_per_model.items()))

    # ------------------------------------------------------------------
    # 6. Coverage at thresholds
    # ------------------------------------------------------------------
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    coverage = {}
    if ai_nn_scores:
        for t in thresholds:
            frac = float(sum(1 for x in ai_nn_scores if x >= t) / len(ai_nn_scores))
            coverage[f'ai_with_human_match_ge_{t:.1f}'] = round(frac, 4)
    if human_nn_scores:
        for t in thresholds:
            frac = float(sum(1 for x in human_nn_scores if x >= t) / len(human_nn_scores))
            coverage[f'human_with_ai_match_ge_{t:.1f}'] = round(frac, 4)

    # ------------------------------------------------------------------
    # 7. Per-paper summary (mean cosine by pair_type, sortable)
    # ------------------------------------------------------------------
    per_paper_rows = []
    for pid in sorted(per_paper.keys()):
        buckets = per_paper[pid]
        row = {
            'paper_id': pid,
            'n_hh': len(buckets['H-H']),
            'n_aa': len(buckets['A-A']),
            'n_ha': len(buckets['H-A']),
            'mean_hh': float(np.mean(buckets['H-H'])) if buckets['H-H'] else None,
            'mean_aa': float(np.mean(buckets['A-A'])) if buckets['A-A'] else None,
            'mean_ha': float(np.mean(buckets['H-A'])) if buckets['H-A'] else None,
        }
        # Gap: how much does an AI item under-match a human item, vs H-H baseline?
        if row['mean_hh'] is not None and row['mean_ha'] is not None:
            row['gap_hh_minus_ha'] = row['mean_hh'] - row['mean_ha']
        per_paper_rows.append(row)

    per_paper_rows.sort(key=lambda r: (r.get('mean_ha') or float('-inf')), reverse=True)
    top_ha = per_paper_rows[:10]
    bottom_ha = sorted(per_paper_rows, key=lambda r: (r.get('mean_ha') or float('inf')))[:10]

    # ------------------------------------------------------------------
    # 8. AI-vs-Human gap test (dataset level)
    # ------------------------------------------------------------------
    # H-H (diff reviewer) is the natural ceiling for "how much agreement is
    # typical between two independent reviewers on the same paper?"
    # Compare H-A directly against that.
    gap = {}
    if hh_diff and ha_scores:
        mu_ceiling = float(np.mean(hh_diff))
        mu_ha = float(np.mean(ha_scores))
        gap = {
            'mean_HH_diff_reviewer (ceiling)': round(mu_ceiling, 4),
            'mean_HA': round(mu_ha, 4),
            'absolute_gap': round(mu_ceiling - mu_ha, 4),
            'percent_of_ceiling': round(100.0 * mu_ha / mu_ceiling, 2) if mu_ceiling > 0 else None,
        }

    # ------------------------------------------------------------------
    # Assemble and write output
    # ------------------------------------------------------------------
    output = {
        'source_pairs_file': str(pairs_path),
        'n_pairs_scored': n_total,
        'global_pair_type_summary': global_summary,
        'same_vs_diff_reviewer_split': split_summary,
        'per_ai_model_human_likeness': per_model_ha,
        'per_ai_model_pair_agreement': per_pair_aa,
        'nearest_neighbor_summary': nn_summary,
        'nearest_neighbor_per_ai_model': nn_per_model,
        'coverage_at_thresholds': coverage,
        'per_paper_top10_highest_mean_HA': top_ha,
        'per_paper_bottom10_lowest_mean_HA': bottom_ha,
        'per_paper_full': per_paper_rows,
        'ai_vs_human_gap': gap,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f'\nWrote analysis → {out_path}')

    # ------------------------------------------------------------------
    # Human-readable printout
    # ------------------------------------------------------------------
    print('\n' + '=' * 72)
    print('  ANALYSIS SUMMARY')
    print('=' * 72)

    def _fmt_sum(s):
        if not s or s.get('n', 0) == 0:
            return f"(n=0)"
        return f"n={s['n']:>5}  mean={s['mean']:+.3f}  median={s['median']:+.3f}  std={s['std']:.3f}"

    print('\n-- Global pair-type --')
    for k in ['H-H', 'A-A', 'H-A']:
        print(f'  {k}: {_fmt_sum(global_summary[k])}')

    print('\n-- Same vs different reviewer --')
    for k, s in split_summary.items():
        print(f'  {k:<22} {_fmt_sum(s)}')

    print('\n-- Per-AI-model human-likeness (higher = closer to humans) --')
    for row in per_model_ha:
        print(f"  {row['label']:<10} mean_HA={row['mean']:+.3f}  median={row['median']:+.3f}  n={row['n']}")

    print('\n-- Per-AI-model-pair agreement (inter-AI) --')
    for row in per_pair_aa:
        print(f"  {row['label']:<26} mean={row['mean']:+.3f}  n={row['n']}")

    print('\n-- Nearest-neighbor summary --')
    print(f"  each AI item → best human match: mean={nn_summary['ai_item_max_to_human'].get('mean', float('nan')):+.3f}  "
          f"median={nn_summary['ai_item_max_to_human'].get('median', float('nan')):+.3f}")
    print(f"  each Human item → best AI match:  mean={nn_summary['human_item_max_to_ai'].get('mean', float('nan')):+.3f}  "
          f"median={nn_summary['human_item_max_to_ai'].get('median', float('nan')):+.3f}")

    print('\n-- Coverage (fraction of items with a cross-type match ≥ τ) --')
    for k, v in coverage.items():
        print(f'  {k}: {v:.3f}')

    if gap:
        print('\n-- AI vs Human gap (H-H_diff is the natural ceiling) --')
        for k, v in gap.items():
            print(f'  {k}: {v}')


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate a full_similarity pairs JSONL into dataset-level statistics.')
    parser.add_argument('pairs_file', type=Path,
                        help='pairs_<backend>.jsonl produced by compute_full_similarity.py')
    parser.add_argument('--items-file', type=Path, default=None,
                        help='items.json (default: sibling of pairs_file)')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output analysis JSON (default: analysis_<backend>.json next to pairs_file)')
    args = parser.parse_args()

    pairs_path = args.pairs_file.resolve()
    items_path = args.items_file.resolve() if args.items_file else pairs_path.parent / 'items.json'
    if args.out is None:
        stem = pairs_path.name
        # pairs_embedding_<safe>.jsonl → analysis_embedding_<safe>.json
        # pairs_<safe>.jsonl            → analysis_<safe>.json  (legacy)
        if stem.startswith('pairs_') and stem.endswith('.jsonl'):
            out_name = 'analysis_' + stem[len('pairs_'):-len('.jsonl')] + '.json'
        else:
            out_name = stem + '.analysis.json'
        out_path = pairs_path.parent / out_name
    else:
        out_path = args.out.resolve()

    analyze(pairs_path, items_path, out_path)


if __name__ == '__main__':
    main()
