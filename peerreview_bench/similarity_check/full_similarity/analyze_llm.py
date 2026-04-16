"""
Aggregate an LLM-judge pairs JSONL produced by compute_full_similarity_llm.py
into paper-level and dataset-level statistics that answer:

    "How similar is AI review compared to human reviews?"

The LLM judge emits a **4-way categorical label** per pair rather than a
continuous cosine, so most of the numbers here are FREQUENCIES (fraction
of pairs in each category) rather than means.

Categories (per prompts.FOURWAY_SYSTEM_PROMPT):
    (c) same subject, same argument, same evidence       — near-paraphrase
    (b) same subject, same argument, different evidence  — convergent
    (a) same subject, different argument                 — topical neighbor
    (d) different subject                                — unrelated

Derived binary label: {c, b} → similar ; {a, d} → not_similar.

Input  : pairs_llm_{slug}.jsonl  (from compute_full_similarity_llm.py)
Output : analysis_llm_{slug}.json — structured report covering:

    1. Global pair-type summary: for each of H-H / A-A / H-A, the
       fraction in each 4-way category plus the "similar" rate.
    2. Same-reviewer vs different-reviewer split within H-H and A-A
       (how often does a reviewer produce similar items to themselves
       vs. to another reviewer?)
    3. Per-AI-model human-likeness: fraction of (H ↔ that model) pairs
       judged similar, and category mix. Ranks AI models.
    4. Per-AI-model-pair agreement (inter-AI): fraction of A-A pairs
       between two specific models judged similar, split by model pair.
    5. Per-paper summary: similar-rate by pair_type. Sortable to find
       the papers where AI comes closest to humans.
    6. AI vs Human gap: compare P(similar | H-A) to P(similar | H-H diff-reviewer),
       which is the natural human-ceiling.
    7. Error / parse-rate audit.

Usage:
    python analyze_llm.py outputs/full_similarity/pairs_llm_<slug>.jsonl
    python analyze_llm.py outputs/full_similarity/pairs_llm_<slug>.jsonl --out analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


# Canonical label set (kept in sync with prompts.py)
FOURWAY_CATS = (
    'same subject, same argument, same evidence',
    'same subject, same argument, different evidence',
    'same subject, different argument',
    'different subject',
)

# Short codes used throughout the paper (matches the audit notation).
LABEL_TO_SHORT = {
    'same subject, same argument, same evidence':      'c',  # near-paraphrase
    'same subject, same argument, different evidence': 'b',  # convergent
    'same subject, different argument':                'a',  # topical neighbor
    'different subject':                               'd',  # unrelated
}

# Labels that map to binary "similar" per the paper's taxonomy.
SIMILAR_LABELS = {
    'same subject, same argument, same evidence',
    'same subject, same argument, different evidence',
}


def _stream_pairs(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f'  warn: dropping truncated line {line_num}', file=sys.stderr)
                continue


def _category_breakdown(labels: List[str]) -> Dict:
    """Return per-category counts + fractions from a list of 4-way labels.
    None / missing labels are tracked as 'unparsed'."""
    counter: Counter = Counter()
    for lab in labels:
        if lab in FOURWAY_CATS:
            counter[lab] += 1
        elif lab is None:
            counter['unparsed'] += 1
        else:
            counter['other'] += 1

    n = len(labels)
    parsed = n - counter.get('unparsed', 0) - counter.get('other', 0)
    result: Dict = {
        'n': n,
        'n_parsed': parsed,
        'parse_rate': round(parsed / n, 4) if n else 0.0,
    }
    # Per-category
    cat_mix: Dict[str, Dict] = {}
    for cat in FOURWAY_CATS:
        c = counter.get(cat, 0)
        cat_mix[LABEL_TO_SHORT[cat]] = {
            'label': cat,
            'n': c,
            'fraction_of_parsed': round(c / parsed, 4) if parsed else 0.0,
            'fraction_of_total': round(c / n, 4) if n else 0.0,
        }
    result['categories'] = cat_mix

    # Binary summary
    n_similar = sum(counter.get(cat, 0) for cat in SIMILAR_LABELS)
    result['n_similar'] = n_similar
    result['similar_rate_of_parsed'] = round(n_similar / parsed, 4) if parsed else 0.0
    result['similar_rate_of_total'] = round(n_similar / n, 4) if n else 0.0
    return result


def analyze(pairs_path: Path, out_path: Path) -> None:
    print(f'Loading pairs from {pairs_path}...')

    # Containers
    hh_labels: List[str] = []
    aa_labels: List[str] = []
    ha_labels: List[str] = []

    hh_same_labels: List[str] = []
    hh_diff_labels: List[str] = []
    aa_same_labels: List[str] = []
    aa_diff_labels: List[str] = []

    per_model_ha_labels: Dict[str, List[str]] = defaultdict(list)
    per_model_pair_aa_labels: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    per_paper: Dict[int, Dict[str, List[str]]] = defaultdict(
        lambda: {'H-H': [], 'A-A': [], 'H-A': []}
    )

    n_total = 0
    n_errors = 0
    errors_by_type: Counter = Counter()

    for rec in _stream_pairs(pairs_path):
        n_total += 1
        lab = rec.get('parsed_answer')
        pt = rec['pair_type']
        pid = rec['paper_id']
        same_rev = rec.get('same_reviewer', False)
        a = rec['item_a']
        b = rec['item_b']

        per_paper[pid][pt].append(lab)

        if rec.get('error'):
            n_errors += 1
            # Truncate the error string for the histogram.
            etype = (rec['error'].split(':')[0] or 'unknown')[:60]
            errors_by_type[etype] += 1

        if pt == 'H-H':
            hh_labels.append(lab)
            (hh_same_labels if same_rev else hh_diff_labels).append(lab)
        elif pt == 'A-A':
            aa_labels.append(lab)
            (aa_same_labels if same_rev else aa_diff_labels).append(lab)
            m1, m2 = sorted([a['reviewer_id'], b['reviewer_id']])
            per_model_pair_aa_labels[(m1, m2)].append(lab)
        elif pt == 'H-A':
            ha_labels.append(lab)
            # The AI side is whichever item has reviewer_type == 'AI'
            ai_side = a if a['reviewer_type'] == 'AI' else b
            per_model_ha_labels[ai_side['reviewer_id']].append(lab)

    print(f'  read {n_total} pair records')
    print(f'  H-H = {len(hh_labels)}, A-A = {len(aa_labels)}, H-A = {len(ha_labels)}')
    print(f'  errors = {n_errors}')

    # ------------------------------------------------------------------
    # 1. Global pair-type summary
    # ------------------------------------------------------------------
    global_summary = {
        'H-H': _category_breakdown(hh_labels),
        'A-A': _category_breakdown(aa_labels),
        'H-A': _category_breakdown(ha_labels),
    }

    # ------------------------------------------------------------------
    # 2. Same-reviewer vs different-reviewer split
    # ------------------------------------------------------------------
    split_summary = {
        'H-H_same_reviewer': _category_breakdown(hh_same_labels),
        'H-H_diff_reviewer': _category_breakdown(hh_diff_labels),
        'A-A_same_model':    _category_breakdown(aa_same_labels),
        'A-A_diff_model':    _category_breakdown(aa_diff_labels),
    }

    # ------------------------------------------------------------------
    # 3. Per-AI-model human-likeness
    # ------------------------------------------------------------------
    per_model_ha = []
    for model in sorted(per_model_ha_labels.keys()):
        row = {'model': model, **_category_breakdown(per_model_ha_labels[model])}
        per_model_ha.append(row)
    per_model_ha.sort(key=lambda r: r.get('similar_rate_of_parsed', 0), reverse=True)

    # ------------------------------------------------------------------
    # 4. Per-AI-model-pair agreement (inter-AI)
    # ------------------------------------------------------------------
    per_pair_aa = []
    for (m1, m2), labs in sorted(per_model_pair_aa_labels.items()):
        label = f'{m1} × {m2}' if m1 != m2 else f'{m1} (same-model)'
        row = {'label': label, 'model_a': m1, 'model_b': m2,
               **_category_breakdown(labs)}
        per_pair_aa.append(row)
    per_pair_aa.sort(key=lambda r: r.get('similar_rate_of_parsed', 0), reverse=True)

    # ------------------------------------------------------------------
    # 5. Per-paper summary
    # ------------------------------------------------------------------
    per_paper_rows = []
    for pid in sorted(per_paper.keys()):
        buckets = per_paper[pid]
        row: Dict = {'paper_id': pid}
        for pt in ('H-H', 'A-A', 'H-A'):
            stats = _category_breakdown(buckets[pt])
            row[f'n_{pt}'] = stats['n']
            row[f'similar_rate_{pt}'] = stats['similar_rate_of_parsed'] if stats['n_parsed'] else None
        per_paper_rows.append(row)
    per_paper_rows.sort(
        key=lambda r: (r.get('similar_rate_H-A') or -1.0),
        reverse=True,
    )
    top_ha = per_paper_rows[:10]
    bottom_ha = sorted(per_paper_rows,
                       key=lambda r: (r.get('similar_rate_H-A') if r.get('similar_rate_H-A') is not None else 2.0))[:10]

    # ------------------------------------------------------------------
    # 6. AI vs Human gap
    # ------------------------------------------------------------------
    ha_sim = global_summary['H-A']['similar_rate_of_parsed']
    hh_diff_sim = split_summary['H-H_diff_reviewer']['similar_rate_of_parsed']
    gap = {
        'P(similar | H-H, different reviewer)  [ceiling]': hh_diff_sim,
        'P(similar | H-A)': ha_sim,
        'absolute_gap': round(hh_diff_sim - ha_sim, 4),
        'percent_of_ceiling': round(100.0 * ha_sim / hh_diff_sim, 2) if hh_diff_sim > 0 else None,
    }

    # ------------------------------------------------------------------
    # 7. Audit
    # ------------------------------------------------------------------
    n_parsed_total = (
        global_summary['H-H']['n_parsed']
        + global_summary['A-A']['n_parsed']
        + global_summary['H-A']['n_parsed']
    )
    audit = {
        'n_pairs_in_file': n_total,
        'n_parsed': n_parsed_total,
        'overall_parse_rate': round(n_parsed_total / n_total, 4) if n_total else 0.0,
        'n_errors': n_errors,
        'error_rate': round(n_errors / n_total, 4) if n_total else 0.0,
        'error_types_top10': errors_by_type.most_common(10),
    }

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------
    output = {
        'source_pairs_file': str(pairs_path),
        'audit': audit,
        'global_pair_type_summary': global_summary,
        'same_vs_diff_reviewer_split': split_summary,
        'per_ai_model_human_likeness': per_model_ha,
        'per_ai_model_pair_agreement': per_pair_aa,
        'per_paper_top10_highest_similar_HA': top_ha,
        'per_paper_bottom10_lowest_similar_HA': bottom_ha,
        'per_paper_full': per_paper_rows,
        'ai_vs_human_gap': gap,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f'\nWrote analysis → {out_path}')

    # ------------------------------------------------------------------
    # Stdout summary
    # ------------------------------------------------------------------
    print('\n' + '=' * 72)
    print('  ANALYSIS SUMMARY (LLM 4-way judge)')
    print('=' * 72)

    def _fmt_cat(cat_block):
        n = cat_block['n']
        sim = cat_block['similar_rate_of_parsed']
        parse = cat_block['parse_rate']
        cats = cat_block['categories']
        return (
            f"n={n:>6}  parsed={parse:.2f}  P(similar)={sim:+.3f}  "
            f"[c={cats['c']['fraction_of_parsed']:.2f} "
            f"b={cats['b']['fraction_of_parsed']:.2f} "
            f"a={cats['a']['fraction_of_parsed']:.2f} "
            f"d={cats['d']['fraction_of_parsed']:.2f}]"
        )

    print('\n-- Global pair-type --')
    for k in ('H-H', 'A-A', 'H-A'):
        print(f'  {k}: {_fmt_cat(global_summary[k])}')

    print('\n-- Same vs different reviewer --')
    for k, s in split_summary.items():
        print(f'  {k:<22} {_fmt_cat(s)}')

    print('\n-- Per-AI-model human-likeness (higher similar-rate = closer to humans) --')
    for row in per_model_ha:
        print(f"  {row['model']:<10} P(similar)={row['similar_rate_of_parsed']:+.3f}  n={row['n']}")

    print('\n-- Per-AI-model-pair agreement --')
    for row in per_pair_aa:
        print(f"  {row['label']:<26} P(similar)={row['similar_rate_of_parsed']:+.3f}  n={row['n']}")

    print('\n-- AI vs Human gap --')
    for k, v in gap.items():
        print(f'  {k}: {v}')

    print('\n-- Parse / error audit --')
    print(f"  total pairs scored: {audit['n_pairs_in_file']}")
    print(f"  parsed: {audit['n_parsed']}  parse_rate: {audit['overall_parse_rate']}")
    print(f"  errors: {audit['n_errors']}  error_rate: {audit['error_rate']}")
    if errors_by_type:
        print('  top error types:')
        for t, c in errors_by_type.most_common(5):
            print(f'    {c:>6}  {t}')


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate a full_similarity LLM-judge pairs JSONL into '
                    'dataset-level 4-way statistics.')
    parser.add_argument('pairs_file', type=Path,
                        help='pairs_llm_<slug>.jsonl from compute_full_similarity_llm.py')
    parser.add_argument('--out', type=Path, default=None,
                        help='Output analysis JSON (default: analysis_llm_<slug>.json '
                             'next to pairs_file)')
    args = parser.parse_args()

    pairs_path = args.pairs_file.resolve()
    if args.out is None:
        stem = pairs_path.name
        if stem.startswith('pairs_') and stem.endswith('.jsonl'):
            out_name = 'analysis_' + stem[len('pairs_'):-len('.jsonl')] + '.json'
        else:
            out_name = stem + '.analysis.json'
        out_path = pairs_path.parent / out_name
    else:
        out_path = args.out.resolve()

    analyze(pairs_path, out_path)


if __name__ == '__main__':
    main()
