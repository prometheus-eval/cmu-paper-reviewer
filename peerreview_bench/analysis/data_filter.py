"""
Shared annotation validity filtering for PeerReview Bench analyses.

This module is imported by the three analysis scripts
(peerreview_analysis.py, peerreview_analysis_per_paper.py,
peerreview_analysis_glmm.py) so they all apply the same data quality rules.

Rules
=====

Annotation cascade
  The PDF annotation workflow is a cascade:
    correctness → significance → evidence
  - Significance is only meaningful if the item is Correct.
  - Evidence is only meaningful if the item is Correct AND
    at least Marginally Significant.

Validity rules
  Rule 2  (not fully annotated):
    Correct but missing significance → drop from ALL analyses.
  Rule 3  (not fully annotated):
    Correct + (Marginally) Significant but missing evidence
    → drop from ALL analyses.
  Rule 5  (overly annotated):
    Not Correct but has a significance annotation
    → drop sig and evidence, keep the item in the correctness analysis.
  Rule 6  (overly annotated):
    Correct + Not Significant but has an evidence annotation
    → drop evidence, keep the item in correctness and significance analyses.

Significance is always 3-class: "Very Significant" is merged into "Significant"
(numeric value 2). See extract_annotations.js.
"""

from typing import Any, Dict, Iterable, List, Set


# ============================================================================
# VALIDITY CATEGORIES
# ============================================================================

# Fully valid categories (included in one or more analyses)
VALID_NOT_CORRECT = 'valid_not_correct'              # Not Correct, no sig/evi
VALID_CORRECT_NOT_SIG = 'valid_correct_not_sig'      # Correct, Not Significant, no evi
VALID_CORRECT_SIG = 'valid_correct_sig'              # Correct, (Marg/)Sig, evi annotated

# "Not fully annotated" — dropped entirely
INVALID_MISSING_SIG = 'invalid_missing_sig'          # Rule 2
INVALID_MISSING_EVI = 'invalid_missing_evi'          # Rule 3

# "Overly annotated" — partial keep
OVERLY_ANNOTATED_SIG = 'overly_annotated_sig'        # Rule 5
OVERLY_ANNOTATED_EVI = 'overly_annotated_evi'        # Rule 6

# Edge case: correctness itself is missing
MISSING_CORRECTNESS = 'missing_correctness'


CATEGORIES_FOR_CORRECTNESS: Set[str] = {
    VALID_NOT_CORRECT,
    VALID_CORRECT_NOT_SIG,
    VALID_CORRECT_SIG,
    OVERLY_ANNOTATED_SIG,   # keep for correctness (rule 5)
    OVERLY_ANNOTATED_EVI,   # keep for correctness (rule 6)
}

CATEGORIES_FOR_SIGNIFICANCE: Set[str] = {
    VALID_CORRECT_NOT_SIG,  # sig = 0 (Not Significant)
    VALID_CORRECT_SIG,      # sig > 0
    OVERLY_ANNOTATED_EVI,   # sig = 0, evi ignored (rule 6)
}

CATEGORIES_FOR_EVIDENCE: Set[str] = {
    VALID_CORRECT_SIG,      # only fully valid items with sig > 0 and evi annotated
}

NOT_FULLY_ANNOTATED_CATEGORIES: Set[str] = {INVALID_MISSING_SIG, INVALID_MISSING_EVI}
OVERLY_ANNOTATED_CATEGORIES: Set[str] = {OVERLY_ANNOTATED_SIG, OVERLY_ANNOTATED_EVI}

# Fully valid categories: items with a clean, cascade-consistent annotation.
# These are the "non-invalid" items for per-reviewer ratio denominators.
FULLY_VALID_CATEGORIES: Set[str] = {
    VALID_NOT_CORRECT, VALID_CORRECT_NOT_SIG, VALID_CORRECT_SIG,
}

# Categories that should be dropped from ALL analyses
FULLY_DROPPED_CATEGORIES: Set[str] = (
    NOT_FULLY_ANNOTATED_CATEGORIES | {MISSING_CORRECTNESS}
)


# ============================================================================
# ITEM CLASSIFICATION
# ============================================================================

def _is_missing(v: Any) -> bool:
    """True if value is None or NaN (works without importing numpy/pandas)."""
    if v is None:
        return True
    # NaN is the only value in Python where x != x is True
    try:
        return v != v  # type: ignore[no-any-return]
    except Exception:
        return False


def _get(item: Any, key: str):
    """Get key from either a dict or an object with attributes."""
    if isinstance(item, dict):
        return item.get(key)
    return getattr(item, key, None)


def classify_item_validity(item: Any) -> str:
    """
    Classify a single item into one of the validity categories above.
    Works on dicts, dataclasses, or any object exposing the required fields.
    Treats None and NaN as "missing".
    """
    corr = _get(item, 'correctness_numeric')
    sig = _get(item, 'significance_numeric')
    evi = _get(item, 'evidence_numeric')

    if _is_missing(corr):
        return MISSING_CORRECTNESS

    if corr == 0:
        # Not Correct
        if not _is_missing(sig):
            return OVERLY_ANNOTATED_SIG  # Rule 5
        return VALID_NOT_CORRECT

    # corr == 1 (Correct)
    if _is_missing(sig):
        return INVALID_MISSING_SIG       # Rule 2

    if sig == 0:
        # Correct + Not Significant
        if not _is_missing(evi):
            return OVERLY_ANNOTATED_EVI  # Rule 6
        return VALID_CORRECT_NOT_SIG

    # Correct + sig > 0 (Marginally/Significant)
    if _is_missing(evi):
        return INVALID_MISSING_EVI       # Rule 3

    return VALID_CORRECT_SIG


def is_fully_good(item: Any) -> bool:
    """
    Item is Correct + Significant (numeric == 2, includes original "Very Significant")
    + Sufficient. Used for the "fully good" per-reviewer summary.
    """
    corr = _get(item, 'correctness_numeric')
    sig = _get(item, 'significance_numeric')
    evi = _get(item, 'evidence_numeric')
    if _is_missing(corr) or _is_missing(sig) or _is_missing(evi):
        return False
    return corr == 1 and sig == 2 and evi == 1


# ============================================================================
# FILTERING
# ============================================================================

def filter_kept_items(items: Iterable) -> List:
    """
    Return items that are kept for analysis: drops items in
    FULLY_DROPPED_CATEGORIES (rules 2 and 3).

    The returned list still contains items in overly-annotated categories —
    those are dropped on a per-metric basis by get_items_for_*.
    """
    return [
        item for item in items
        if classify_item_validity(item) not in FULLY_DROPPED_CATEGORIES
    ]


def get_items_for_correctness(items: Iterable) -> List:
    return [i for i in items if classify_item_validity(i) in CATEGORIES_FOR_CORRECTNESS]


def get_items_for_significance(items: Iterable) -> List:
    return [i for i in items if classify_item_validity(i) in CATEGORIES_FOR_SIGNIFICANCE]


def get_items_for_evidence(items: Iterable) -> List:
    return [i for i in items if classify_item_validity(i) in CATEGORIES_FOR_EVIDENCE]


# ============================================================================
# DATA QUALITY SUMMARY
# ============================================================================

def summarize_data_quality(items: Iterable) -> Dict:
    """Return a dict with item counts at each validity stage."""
    items = list(items)

    cat_counts = {
        MISSING_CORRECTNESS: 0,
        VALID_NOT_CORRECT: 0,
        VALID_CORRECT_NOT_SIG: 0,
        VALID_CORRECT_SIG: 0,
        INVALID_MISSING_SIG: 0,
        INVALID_MISSING_EVI: 0,
        OVERLY_ANNOTATED_SIG: 0,
        OVERLY_ANNOTATED_EVI: 0,
    }
    for item in items:
        cat_counts[classify_item_validity(item)] += 1

    n_not_fully = cat_counts[INVALID_MISSING_SIG] + cat_counts[INVALID_MISSING_EVI]
    n_overly = cat_counts[OVERLY_ANNOTATED_SIG] + cat_counts[OVERLY_ANNOTATED_EVI]
    n_missing_corr = cat_counts[MISSING_CORRECTNESS]
    n_kept = sum(
        cat_counts[k] for k in (
            VALID_NOT_CORRECT, VALID_CORRECT_NOT_SIG, VALID_CORRECT_SIG,
            OVERLY_ANNOTATED_SIG, OVERLY_ANNOTATED_EVI,
        )
    )

    return {
        'total_raw': len(items),
        'categories': cat_counts,
        'not_fully_annotated_total': n_not_fully,
        'overly_annotated_total': n_overly,
        'missing_correctness_total': n_missing_corr,
        'kept_for_analysis': n_kept,
        'n_for_correctness': len(get_items_for_correctness(items)),
        'n_for_significance': len(get_items_for_significance(items)),
        'n_for_evidence': len(get_items_for_evidence(items)),
    }


# ============================================================================
# ANNOTATOR COUNTS (hard-coded)
#
# The HuggingFace dataset anonymizes reviewer ids (GPT, Claude, Gemini,
# Human_1/2/3), so counting unique reviewer_ids on the dataset only gives 6.
# The real number of meta-reviewers (annotators who graded review items)
# is hard-coded here: 30 primary annotators + 16 secondary annotators, with
# 1 annotator (Yong Jeong) serving as both primary and secondary.
# Union = 45 distinct human annotators. Update these numbers if new
# annotators are added upstream.
# ============================================================================

N_PRIMARY_ANNOTATORS = 30
N_SECONDARY_ANNOTATORS = 16
N_ANNOTATORS_BOTH_ROLES = 1
N_TOTAL_ANNOTATORS = (
    N_PRIMARY_ANNOTATORS + N_SECONDARY_ANNOTATORS - N_ANNOTATORS_BOTH_ROLES
)  # = 45


def summarize_dataset(items: Iterable) -> Dict:
    """Return a coverage summary for a list of ReviewItem objects:
    number of papers, total items, how many items have a non-null label on
    each axis (after validity stripping), and a cascade-tree breakdown
    showing the joint distribution of correctness → significance → evidence.

    `n_annotators` is the hard-coded total because the HF dataset
    anonymizes reviewer ids."""
    items = list(items)
    papers = set()
    n_correctness = 0
    n_significance = 0
    n_evidence = 0

    # Cascade buckets
    n_correct = 0
    n_incorrect = 0
    n_sig_significant = 0
    n_sig_marginal = 0
    n_sig_notsig = 0
    n_sigSuf = 0
    n_sigInsuf = 0
    n_margSuf = 0
    n_margInsuf = 0

    for item in items:
        papers.add(_get(item, 'paper_id'))
        cat = classify_item_validity(item)
        if cat in FULLY_DROPPED_CATEGORIES:
            continue

        corr = _get(item, 'correctness_numeric')
        sig = _get(item, 'significance_numeric')
        evi = _get(item, 'evidence_numeric')

        if cat in (VALID_NOT_CORRECT, VALID_CORRECT_NOT_SIG, VALID_CORRECT_SIG,
                   OVERLY_ANNOTATED_SIG, OVERLY_ANNOTATED_EVI):
            n_correctness += 1
            # Cascade split
            if corr == 1:
                n_correct += 1
                if sig == 2:
                    n_sig_significant += 1
                    if evi == 1:
                        n_sigSuf += 1
                    elif evi == 0:
                        n_sigInsuf += 1
                elif sig == 1:
                    n_sig_marginal += 1
                    if evi == 1:
                        n_margSuf += 1
                    elif evi == 0:
                        n_margInsuf += 1
                elif sig == 0:
                    n_sig_notsig += 1
            elif corr == 0:
                n_incorrect += 1
        if cat in (VALID_CORRECT_NOT_SIG, VALID_CORRECT_SIG, OVERLY_ANNOTATED_EVI):
            n_significance += 1
        if cat == VALID_CORRECT_SIG:
            n_evidence += 1

    return {
        'n_papers': len(papers),
        'n_annotators': N_TOTAL_ANNOTATORS,
        'n_primary_annotators': N_PRIMARY_ANNOTATORS,
        'n_secondary_annotators': N_SECONDARY_ANNOTATORS,
        'n_annotators_both_roles': N_ANNOTATORS_BOTH_ROLES,
        'n_review_items': n_correctness,
        'n_correctness_annotations': n_correctness,
        'n_significance_annotations': n_significance,
        'n_evidence_annotations': n_evidence,
        'cascade': {
            'total': n_correctness,
            'correct': n_correct,
            'incorrect': n_incorrect,
            'sig_significant': n_sig_significant,
            'sig_marginal': n_sig_marginal,
            'sig_notsig': n_sig_notsig,
            'sig_sig_sufficient': n_sigSuf,
            'sig_sig_insufficient': n_sigInsuf,
            'sig_marg_sufficient': n_margSuf,
            'sig_marg_insufficient': n_margInsuf,
        },
    }


def format_dataset_overview(summary: Dict) -> str:
    """Format the dataset overview as a text block. Includes a tree-style
    visualization of the correctness → significance → evidence cascade."""
    c = summary['cascade']
    total = c['total']
    lines = []
    lines.append("=" * 100)
    lines.append("DATASET OVERVIEW")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  Papers covered:                    {summary['n_papers']}")
    lines.append(
        f"  Annotators (total unique):         {summary['n_annotators']}  "
        f"(primary: {summary['n_primary_annotators']}, "
        f"secondary: {summary['n_secondary_annotators']}, "
        f"both roles: {summary['n_annotators_both_roles']})"
    )
    lines.append(f"  Review items:                      {summary['n_review_items']}")
    lines.append("")
    lines.append("  Annotations gathered per axis:")
    lines.append(f"    Correctness:   {summary['n_correctness_annotations']}")
    lines.append(f"    Significance:  {summary['n_significance_annotations']}")
    lines.append(f"    Evidence:      {summary['n_evidence_annotations']}")
    lines.append("")
    lines.append("  Cascade (correctness → significance → evidence):")
    lines.append("")

    def _pct(n: int, denom: int) -> str:
        return f"{n / denom * 100:.2f}%" if denom else "n/a"

    def _of_all(n: int) -> str:
        return _pct(n, total)

    lines.append(f"  All items ({total})")
    lines.append(
        f"  ├── Correct                   {c['correct']:>5}  "
        f"({_of_all(c['correct'])} of all)"
    )
    lines.append(
        f"  │   ├── Significant           {c['sig_significant']:>5}  "
        f"({_pct(c['sig_significant'], c['correct'])} of correct; "
        f"{_of_all(c['sig_significant'])} of all)"
    )
    lines.append(
        f"  │   │   ├── Sufficient        {c['sig_sig_sufficient']:>5}  "
        f"({_pct(c['sig_sig_sufficient'], c['sig_significant'])} of significant; "
        f"{_of_all(c['sig_sig_sufficient'])} of all)"
    )
    lines.append(
        f"  │   │   └── Not Sufficient    {c['sig_sig_insufficient']:>5}  "
        f"({_pct(c['sig_sig_insufficient'], c['sig_significant'])} of significant; "
        f"{_of_all(c['sig_sig_insufficient'])} of all)"
    )
    lines.append(
        f"  │   ├── Marginally Sig.       {c['sig_marginal']:>5}  "
        f"({_pct(c['sig_marginal'], c['correct'])} of correct; "
        f"{_of_all(c['sig_marginal'])} of all)"
    )
    lines.append(
        f"  │   │   ├── Sufficient        {c['sig_marg_sufficient']:>5}  "
        f"({_pct(c['sig_marg_sufficient'], c['sig_marginal'])} of marginally; "
        f"{_of_all(c['sig_marg_sufficient'])} of all)"
    )
    lines.append(
        f"  │   │   └── Not Sufficient    {c['sig_marg_insufficient']:>5}  "
        f"({_pct(c['sig_marg_insufficient'], c['sig_marginal'])} of marginally; "
        f"{_of_all(c['sig_marg_insufficient'])} of all)"
    )
    lines.append(
        f"  │   └── Not Significant       {c['sig_notsig']:>5}  "
        f"({_pct(c['sig_notsig'], c['correct'])} of correct; "
        f"{_of_all(c['sig_notsig'])} of all)"
    )
    lines.append(
        f"  └── Incorrect                 {c['incorrect']:>5}  "
        f"({_of_all(c['incorrect'])} of all)"
    )
    lines.append("")
    return "\n".join(lines)


# ============================================================================
# "FULLY GOOD" STATS PER REVIEWER GROUP
# ============================================================================

def compute_fully_good_stats(groups: Dict[str, List]) -> Dict[str, Dict]:
    """
    For each group in `groups`, compute:
      - n_fully_good: items that are Correct + Significant (numeric 2) + Sufficient
      - n_valid:      all items by the reviewer (post cascade-stripping)
      - n_papers:     distinct papers the group has items on
      - ratio:        n_fully_good / n_valid
      - ci_lower/ci_upper: Wilson 95% CI on `ratio`
    Assumes the caller has already run `filter_kept_items` upstream.
    """
    from statsmodels.stats.proportion import proportion_confint  # local import
    out: Dict[str, Dict] = {}
    for name, items in groups.items():
        n_valid = len(items)
        n_fg = sum(1 for i in items if is_fully_good(i))
        ratio = (n_fg / n_valid) if n_valid > 0 else 0.0
        if n_valid > 0:
            ci_lo, ci_hi = proportion_confint(n_fg, n_valid, alpha=0.05, method='wilson')
        else:
            ci_lo, ci_hi = float('nan'), float('nan')
        n_papers = len({_get(i, 'paper_id') for i in items})
        out[name] = {
            'n_fully_good': n_fg,
            'n_valid': n_valid,
            'n_papers': n_papers,
            'ratio': ratio,
            'ci_lower': float(ci_lo),
            'ci_upper': float(ci_hi),
        }
    return out


def format_fully_good_report(group_stats: Dict[str, Dict]) -> str:
    """Pretty-print per-group fully-good stats, including 95% CI on the ratio
    and the number of distinct papers contributing."""
    lines = []
    lines.append("=" * 100)
    lines.append("FULLY-GOOD ITEMS PER REVIEWER GROUP")
    lines.append("  'Fully good' = Correct + Significant + Sufficient Evidence")
    lines.append("  95% CI on the ratio is computed via the Wilson score interval.")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"  {'Group':<15} {'Fully Good':>12} {'Total Items':>13} {'(# papers)':>12} "
                 f"{'Ratio':>9} {'95% CI':>20}")
    lines.append("  " + "-" * 85)
    for name, s in group_stats.items():
        if s['n_valid'] > 0 and not math.isnan(s['ci_lower']):
            ci_str = f"[{s['ci_lower']:.1%}, {s['ci_upper']:.1%}]"
        else:
            ci_str = 'N/A'
        lines.append(
            f"  {name:<15} {s['n_fully_good']:>12} {s['n_valid']:>13} "
            f"{'(' + str(s['n_papers']) + ')':>12} {s['ratio']:>8.1%} {ci_str:>20}"
        )
    lines.append("")
    return "\n".join(lines)


import math  # noqa: E402  (late import for format_fully_good_report only)
