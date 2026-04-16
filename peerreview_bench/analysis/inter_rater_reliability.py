#!/usr/bin/env python3
"""
PeerReview Bench - Inter-Rater Reliability (IRR)

Compares primary-reviewer and secondary-reviewer annotations on the overlap
papers and reports two chance-corrected agreement coefficients side by side:

  - Cohen's κ        (baseline, sensitive to marginal distributions)
  - Gwet's AC1       (more robust to class imbalance; recommended alongside κ)

The main output is a single per-axis × per-reviewer-type table showing:
  metric, reviewer type, N, percent agreement, κ, κ label, AC1, AC1 label
with a combined "Overall" section that collapses Human + AI together.

The report also shows the 10-class meta_reviewer label distribution for the
27 overlap papers (the "joint outcome" of (primary, secondary)), loaded from
the `meta_reviewer` HF config.

Outputs:
  - irr_report.txt        human-readable summary with both tables
  - irr_item_level.csv    per-item comparison (for auditing disagreements)
  - irr_summary.json      programmatic summary

Usage:
    python inter_rater_reliability.py --output-dir <output_dir>
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Allow `load_data` imports from the parent peerreview_bench/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from load_data import load_annotations, load_meta_reviewer

warnings.filterwarnings('ignore')


# ============================================================================
# LOAD ITEMS AS A DATAFRAME
# ============================================================================

def load_items_df(annotator_source: str) -> pd.DataFrame:
    items, _ = load_annotations(annotator_source=annotator_source)
    rows = [{
        'paper_id': i.paper_id,
        'reviewer_id': i.reviewer_id,
        'reviewer_type': i.reviewer_type,
        'model_name': i.model_name,
        'item_number': i.item_number,
        'correctness': i.correctness,
        'significance': i.significance,
        'evidence': i.evidence,
    } for i in items]
    return pd.DataFrame(rows)


def merge_primary_secondary(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    """Inner-merge on (paper_id, reviewer_id, item_number). Each output row
    has both primary and secondary annotations as _1 and _2 suffixed cols."""
    key_cols = ['paper_id', 'reviewer_id', 'item_number']
    value_cols = ['reviewer_type', 'model_name', 'correctness', 'significance', 'evidence']
    p = primary[key_cols + value_cols].rename(columns={c: f'{c}_1' for c in value_cols})
    s = secondary[key_cols + value_cols].rename(columns={c: f'{c}_2' for c in value_cols})
    return p.merge(s, on=key_cols, how='inner')


# ============================================================================
# AGREEMENT METRICS: κ + Gwet's AC1
# ============================================================================

def percent_agreement(a: pd.Series, b: pd.Series) -> Tuple[float, int]:
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n == 0:
        return float('nan'), 0
    return float((a[mask] == b[mask]).mean()), n


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return float('nan')
    a_vals = a[mask]
    b_vals = b[mask]
    if a_vals.nunique() < 2 and b_vals.nunique() < 2 and (a_vals.iloc[0] == b_vals.iloc[0]):
        return float('nan')
    try:
        return float(cohen_kappa_score(a_vals, b_vals))
    except Exception:
        return float('nan')


def gwet_ac1(a: pd.Series, b: pd.Series) -> float:
    """Gwet's AC1 chance-corrected agreement coefficient.

        AC1 = (P_obs - P_e_γ) / (1 - P_e_γ)

    where P_obs is the observed agreement, K is the number of distinct
    categories actually used by either rater, and

        P_e_γ = (1 / (K-1)) * Σ_k π_k * (1 - π_k)

    with π_k the marginal probability of category k averaged across the two
    raters. This differs from Cohen's κ — Gwet's AC1 is much less sensitive
    to class imbalance and to the "kappa paradox" where high percent
    agreement produces low κ when one category dominates.
    """
    mask = a.notna() & b.notna()
    n = int(mask.sum())
    if n == 0:
        return float('nan')
    a_vals = list(a[mask])
    b_vals = list(b[mask])

    categories = sorted(set(a_vals) | set(b_vals))
    K = len(categories)
    if K < 2:
        return float('nan')

    p_obs = sum(1 for x, y in zip(a_vals, b_vals) if x == y) / n

    # Marginal probability of each category averaged across raters
    counts: Dict = {c: 0 for c in categories}
    for v in a_vals:
        counts[v] += 1
    for v in b_vals:
        counts[v] += 1
    pi_k = {c: counts[c] / (2 * n) for c in categories}

    p_e_gamma = sum(pi_k[c] * (1 - pi_k[c]) for c in categories) / (K - 1)
    if p_e_gamma >= 1.0:
        return float('nan')
    return float((p_obs - p_e_gamma) / (1 - p_e_gamma))


def interpret_agreement(coef: float) -> str:
    """Landis & Koch (1977) interpretation labels. Used for both κ and AC1."""
    if np.isnan(coef):
        return 'undefined'
    if coef < 0:
        return 'poor'
    if coef < 0.20:
        return 'slight'
    if coef < 0.40:
        return 'fair'
    if coef < 0.60:
        return 'moderate'
    if coef < 0.80:
        return 'substantial'
    return 'almost perfect'


# ============================================================================
# PER-(METRIC, REVIEWER-TYPE) COMPUTATION
# ============================================================================

def _compute_row(sub: pd.DataFrame, metric: str) -> Dict:
    a = sub[f'{metric}_1']
    b = sub[f'{metric}_2']
    pct, n = percent_agreement(a, b)
    k = cohen_kappa(a, b)
    ac1 = gwet_ac1(a, b)
    return {
        'metric': metric,
        'n': n,
        'percent_agreement': pct,
        'kappa': k,
        'kappa_label': interpret_agreement(k),
        'ac1': ac1,
        'ac1_label': interpret_agreement(ac1),
    }


def compute_irr_table(merged: pd.DataFrame) -> List[Dict]:
    """Build the row list for the per-reviewer-type + combined table."""
    rows: List[Dict] = []
    for metric in ('correctness', 'significance', 'evidence'):
        for rtype_label, mask in (
            ('Human', merged['reviewer_type_1'] == 'Human'),
            ('AI', merged['reviewer_type_1'] == 'AI'),
        ):
            sub = merged[mask]
            row = _compute_row(sub, metric)
            row['reviewer_type'] = rtype_label
            rows.append(row)
    # Overall (combined Human + AI)
    for metric in ('correctness', 'significance', 'evidence'):
        row = _compute_row(merged, metric)
        row['reviewer_type'] = 'All'
        row['_combined'] = True
        rows.append(row)
    return rows


# ============================================================================
# META_REVIEWER LABEL DISTRIBUTION
# ============================================================================

META_LABEL_ORDER = [
    'correct_significant_sufficient',
    'correct_significant_insufficient',
    'correct_significant_disagree_on_evidence',
    'correct_marginal_sufficient',
    'correct_marginal_insufficient',
    'correct_marginal_disagree_on_evidence',
    'correct_not_significant',
    'correct_disagree_on_significance',
    'incorrect',
    'disagree_on_correctness',
]


def compute_meta_label_distribution() -> Optional[List[Dict]]:
    """Load the meta_reviewer HF config and count items per 10-class label."""
    try:
        rows = load_meta_reviewer()
    except Exception as e:
        print(f"  (could not load meta_reviewer config: {e})", file=sys.stderr)
        return None
    from collections import Counter
    counter = Counter(r.get('label') for r in rows)
    total = sum(counter.values()) or 1
    out = []
    for i, label in enumerate(META_LABEL_ORDER, start=1):
        count = counter.get(label, 0)
        out.append({
            'id': i,
            'label': label,
            'count': count,
            'pct': count / total,
        })
    return out


# ============================================================================
# REPORTING
# ============================================================================

def _fmt_table(rows: List[Dict]) -> str:
    lines = []
    # Column widths tuned for monospaced display
    lines.append(
        f"{'Metric':<14} {'Reviewer type':<16} {'N':>5}  "
        f"{'Percent agreement':>18}  {'κ':>6} {'κ label':<16}  "
        f"{'AC1':>6} {'AC1 label':<16}"
    )
    lines.append("-" * 108)
    def _fmt(row):
        pct = f"{row['percent_agreement']*100:.1f}%" if not np.isnan(row['percent_agreement']) else 'N/A'
        k = f"{row['kappa']:.2f}" if not np.isnan(row['kappa']) else 'N/A'
        ac1 = f"{row['ac1']:.2f}" if not np.isnan(row['ac1']) else 'N/A'
        return (
            f"{row['metric'].capitalize():<14} {row['reviewer_type']:<16} {row['n']:>5}  "
            f"{pct:>18}  {k:>6} {row['kappa_label']:<16}  "
            f"{ac1:>6} {row['ac1_label']:<16}"
        )
    # Per-reviewer-type rows first
    for row in rows:
        if row.get('_combined'):
            continue
        lines.append(_fmt(row))
    # Separator and combined section
    lines.append("")
    lines.append("Overall (combined)")
    lines.append("-" * 108)
    for row in rows:
        if not row.get('_combined'):
            continue
        lines.append(_fmt(row))
    return "\n".join(lines)


def _fmt_meta_distribution(dist: Optional[List[Dict]]) -> str:
    if not dist:
        return "  (meta_reviewer config not available — skipping label distribution table)"
    lines = []
    lines.append(f"{'#':>3}  {'Label':<45} {'Count':>7} {'%':>7}")
    lines.append("-" * 70)
    total = sum(d['count'] for d in dist)
    for d in dist:
        lines.append(
            f"{d['id']:>3}  {d['label']:<45} {d['count']:>7} {d['pct']*100:>6.1f}%"
        )
    lines.append("-" * 70)
    lines.append(f"{'Total':>3}  {'':<45} {total:>7} {'100.0%':>7}")
    return "\n".join(lines)


def build_report(merged: pd.DataFrame, rows: List[Dict],
                 meta_dist: Optional[List[Dict]]) -> str:
    lines = []
    lines.append("=" * 108)
    lines.append("PEERREVIEW BENCH — INTER-RATER RELIABILITY REPORT")
    lines.append("=" * 108)
    lines.append("")
    lines.append(f"  Shared items compared: {len(merged)}")
    lines.append(f"  Papers covered:        {merged['paper_id'].nunique()}")
    lines.append(f"  Unique reviewers:      {merged['reviewer_id'].nunique()}")
    total_reviewer_types = merged['reviewer_type_1'].value_counts().to_dict()
    lines.append("  By reviewer type:")
    for k, v in sorted(total_reviewer_types.items()):
        lines.append(f"    {k:<8} {v} items")
    lines.append("")

    lines.append("AGREEMENT INTERPRETATION (Landis & Koch 1977, same labels used for κ and AC1):")
    lines.append("  ≤ 0.00  : poor")
    lines.append("  0.00 – 0.20 : slight")
    lines.append("  0.21 – 0.40 : fair")
    lines.append("  0.41 – 0.60 : moderate")
    lines.append("  0.61 – 0.80 : substantial")
    lines.append("  0.81 – 1.00 : almost perfect")
    lines.append("")
    lines.append("Note on Gwet's AC1: more robust to class imbalance than Cohen's κ.")
    lines.append("When one category dominates, κ can understate agreement (the 'kappa paradox').")
    lines.append("AC1 does not suffer from this and tends to be the better single-number summary.")
    lines.append("")

    lines.append("=" * 108)
    lines.append("AGREEMENT TABLE")
    lines.append("=" * 108)
    lines.append(_fmt_table(rows))
    lines.append("")
    lines.append("")
    lines.append("=" * 108)
    lines.append("META-REVIEWER LABEL DISTRIBUTION (10-class joint outcome)")
    lines.append("=" * 108)
    lines.append("Source: `meta_reviewer` HF config. Each item is one of the 10 classes that")
    lines.append("encode the joint primary/secondary cascade outcome plus per-axis agreement.")
    lines.append("")
    lines.append(_fmt_meta_distribution(meta_dist))
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='PeerReview Bench IRR')
    parser.add_argument('--output-dir', type=str, default='./irr_output')
    args = parser.parse_args()

    print("Loading annotations from HuggingFace dataset...")
    primary = load_items_df('primary')
    secondary = load_items_df('secondary')
    print(f"  Primary:   {len(primary):>5} items, {primary['paper_id'].nunique()} papers")
    print(f"  Secondary: {len(secondary):>5} items, {secondary['paper_id'].nunique()} papers")

    merged = merge_primary_secondary(primary, secondary)
    if len(merged) == 0:
        print("No overlapping items between primary and secondary annotations.")
        print("Check that reviewer_id and item_number align between the two sets.")
        return
    print(f"  Overlap:   {len(merged):>5} items across {merged['paper_id'].nunique()} papers")

    rows = compute_irr_table(merged)
    print("\nLoading meta_reviewer config for label distribution...")
    meta_dist = compute_meta_label_distribution()

    report = build_report(merged, rows, meta_dist)
    print(report)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / 'irr_report.txt', 'w') as f:
        f.write(report)

    # Per-item comparison CSV (for auditing disagreements)
    item_level = merged[[
        'paper_id', 'reviewer_id', 'reviewer_type_1', 'model_name_1', 'item_number',
        'correctness_1', 'correctness_2',
        'significance_1', 'significance_2',
        'evidence_1', 'evidence_2',
    ]].rename(columns={
        'reviewer_type_1': 'reviewer_type',
        'model_name_1': 'model_name',
        'correctness_1': 'correctness_primary',  'correctness_2': 'correctness_secondary',
        'significance_1': 'significance_primary', 'significance_2': 'significance_secondary',
        'evidence_1': 'evidence_primary',        'evidence_2': 'evidence_secondary',
    })
    item_level['correctness_agree'] = item_level['correctness_primary'] == item_level['correctness_secondary']
    item_level['significance_agree'] = item_level['significance_primary'] == item_level['significance_secondary']
    item_level['evidence_agree'] = item_level['evidence_primary'] == item_level['evidence_secondary']
    item_level.to_csv(out / 'irr_item_level.csv', index=False)

    json_safe = {
        'agreement_table': [
            {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in r.items()}
            for r in rows
        ],
        'meta_reviewer_label_distribution': meta_dist,
        'n_overlap_items': int(len(merged)),
        'n_overlap_papers': int(merged['paper_id'].nunique()),
    }
    with open(out / 'irr_summary.json', 'w') as f:
        json.dump(json_safe, f, indent=2,
                  default=lambda x: None if isinstance(x, float) and np.isnan(x) else x)

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
