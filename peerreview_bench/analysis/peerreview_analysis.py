#!/usr/bin/env python3
"""
PeerReview Bench - Primary Statistical Analysis (PAIRED)

Paired paper-level comparisons across 5 reviewer groups:
    Best Human, Worst Human, GPT, Claude, Gemini.

- Descriptive stats: item-level (transparent reporting)
- Inference: paired paper-level differences (raw p-values, no multiple-comparison correction)
- Binary metrics (correctness, evidence): paired t-test, Cohen's d
- Ordinal metric (significance, 0-2 scale): Wilcoxon signed-rank, rank-biserial r
- Classification uses metric-appropriate effect size gate

Usage:
    python peerreview_analysis.py --data-dir <json_dir> --output-dir <output_dir> --rankings-file <rankings.json>
    python peerreview_analysis.py --combined <combined.json> --output-dir <output_dir> --rankings-file <rankings.json>
"""

import json
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Allow `load_data` imports from the parent peerreview_bench/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_1samp, wilcoxon
from statsmodels.stats.proportion import proportion_confint

from data_filter import (
    CATEGORIES_FOR_CORRECTNESS, CATEGORIES_FOR_SIGNIFICANCE, CATEGORIES_FOR_EVIDENCE,
    classify_item_validity, is_fully_good,
    filter_kept_items,
    get_items_for_correctness as _filter_for_correctness,
    get_items_for_significance as _filter_for_significance,
    get_items_for_evidence as _filter_for_evidence,
    summarize_dataset, format_dataset_overview,
    compute_fully_good_stats, format_fully_good_report,
)
from load_data import ReviewItem, load_annotations

warnings.filterwarnings('ignore')

N_BOOTSTRAP = 10000
RANDOM_SEED = 42

# ============================================================================
# CONFIGURATION
# ============================================================================

ALPHA = 0.05

COHENS_D_THRESHOLDS = {
    'negligible': 0.20,
    'small': 0.50,
    'medium': 0.80,
}

# Rank-biserial correlation thresholds (Kerby 2014)
RANK_BISERIAL_THRESHOLDS = {
    'negligible': 0.10,
    'small': 0.30,
    'medium': 0.50,
}


# ============================================================================
# CONFIDENCE INTERVAL HELPERS
# ============================================================================

def wilson_ci(count: int, nobs: int, alpha: float = 0.05) -> Tuple[float, float]:
    if nobs == 0:
        return (np.nan, np.nan)
    return proportion_confint(count, nobs, alpha=alpha, method='wilson')


def bootstrap_ci_mean(values: List[int], alpha: float = 0.05, n_bootstrap: int = N_BOOTSTRAP) -> Tuple[float, float]:
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(RANDOM_SEED)
    values_arr = np.array(values)
    n = len(values_arr)
    bootstrap_means = [np.mean(rng.choice(values_arr, size=n, replace=True)) for _ in range(n_bootstrap)]
    return (np.percentile(bootstrap_means, 100 * alpha / 2),
            np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PairedComparisonResult:
    group1: str
    group2: str
    metric: str
    # Item-level descriptive
    n1_items: int
    n2_items: int
    pos1_items: float
    pos2_items: float
    rate1_items: float
    rate2_items: float
    rate1_ci_lower: float
    rate1_ci_upper: float
    rate2_ci_lower: float
    rate2_ci_upper: float
    # Paper-level inference
    n1_papers: int
    n2_papers: int
    n_paired_papers: int
    mean_diff: float
    std_diff: float
    se_diff: float
    ci_diff_lower: float
    ci_diff_upper: float
    test_statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_size_name: str
    effect_magnitude: str
    direction: str
    classification: str


# ============================================================================
# DATA LOADING
# ============================================================================

# ============================================================================
# GROUP CONSTRUCTION
# ============================================================================

def create_comparison_groups(
    items: List[ReviewItem],
    reviewer_rankings: Dict[int, Dict[str, str]],
) -> Dict[str, List[ReviewItem]]:
    """Create the 5 comparison groups.

    Special case: on papers with a single human reviewer, the rankings file
    encodes `best == worst == <that reviewer>`. In that case the reviewer is
    trivially both the best and the worst available human, so we add the
    item to BOTH groups. This keeps Best Human and Worst Human at identical
    paper coverage (important for cross-paper aggregates). The downstream
    Best vs Worst paired comparison still works: such papers contribute a
    zero paired difference (same items on both sides), Wilcoxon drops zero
    ties from its ranking, and the t-test is essentially unaffected given
    the small number of single-reviewer papers.
    """
    groups = {'Best Human': [], 'Worst Human': [], 'GPT': [], 'Claude': [], 'Gemini': []}

    for item in items:
        paper_id = item.paper_id
        if item.reviewer_type == 'Human':
            if paper_id in reviewer_rankings:
                r = reviewer_rankings[paper_id]
                if item.reviewer_id == r.get('best'):
                    groups['Best Human'].append(item)
                # NOTE: intentionally not elif — when best == worst, the
                # same item should land in both groups.
                if item.reviewer_id == r.get('worst'):
                    groups['Worst Human'].append(item)
        elif item.reviewer_type == 'AI':
            if item.model_name in groups:
                groups[item.model_name].append(item)
    return groups


# ============================================================================
# CASCADING FILTERS
# ============================================================================

def get_items_for_correctness(items: List[ReviewItem]) -> List[ReviewItem]:
    """Items contributing to correctness analysis (per data_filter rules)."""
    return _filter_for_correctness(items)


def get_items_for_significance(items: List[ReviewItem]) -> List[ReviewItem]:
    """Items contributing to significance analysis (per data_filter rules)."""
    return _filter_for_significance(items)


def get_items_for_evidence(items: List[ReviewItem]) -> List[ReviewItem]:
    """Items contributing to evidence analysis (per data_filter rules)."""
    return _filter_for_evidence(items)


def extract_metric_values(items: List[ReviewItem], metric: str) -> Tuple[List[int], float, int]:
    if metric == 'correctness':
        filtered = get_items_for_correctness(items)
        values = [i.correctness_numeric for i in filtered]
    elif metric == 'significance':
        filtered = get_items_for_significance(items)
        values = [i.significance_numeric for i in filtered]
    else:
        filtered = get_items_for_evidence(items)
        values = [i.evidence_numeric for i in filtered]

    total = len(values)
    positive = sum(values) if values else 0
    return values, positive, total


# ============================================================================
# PAPER-LEVEL RATES
# ============================================================================

def calculate_paper_rates(items: List[ReviewItem], metric: str) -> Dict[int, float]:
    """Per-paper rate (binary) or mean ordinal score (significance)."""
    if metric == 'correctness':
        filtered = get_items_for_correctness(items)
        get_v = lambda i: i.correctness_numeric
    elif metric == 'significance':
        filtered = get_items_for_significance(items)
        get_v = lambda i: i.significance_numeric
    else:
        filtered = get_items_for_evidence(items)
        get_v = lambda i: i.evidence_numeric

    by_paper: Dict[int, List[int]] = {}
    for item in filtered:
        by_paper.setdefault(item.paper_id, []).append(get_v(item))

    paper_rates = {}
    for pid, vals in by_paper.items():
        total = len(vals)
        if total == 0:
            paper_rates[pid] = np.nan
        elif metric == 'significance':
            paper_rates[pid] = float(np.mean(vals))
        else:
            paper_rates[pid] = sum(vals) / total
    return paper_rates


# ============================================================================
# EFFECT SIZES
# ============================================================================

def _magnitude_from_thresholds(abs_val: float, thresholds: Dict[str, float]) -> str:
    if abs_val < thresholds['negligible']:
        return 'negligible'
    if abs_val < thresholds['small']:
        return 'small'
    if abs_val < thresholds['medium']:
        return 'medium'
    return 'large'


def cohens_d_paired(differences: np.ndarray) -> Tuple[float, str]:
    n = len(differences)
    if n < 2:
        return np.nan, 'undetermined'
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    if std_diff == 0:
        if mean_diff == 0:
            return 0.0, 'negligible'
        return (np.inf if mean_diff > 0 else -np.inf), 'large'
    d = mean_diff / std_diff
    return d, _magnitude_from_thresholds(abs(d), COHENS_D_THRESHOLDS)


def rank_biserial_correlation(differences: np.ndarray) -> Tuple[float, str]:
    nonzero = differences[differences != 0]
    if len(nonzero) < 1:
        return 0.0, 'negligible'
    ranks = stats.rankdata(np.abs(nonzero))
    w_plus = np.sum(ranks[nonzero > 0])
    w_minus = np.sum(ranks[nonzero < 0])
    total = w_plus + w_minus
    if total == 0:
        return 0.0, 'negligible'
    r = (w_plus - w_minus) / total
    return r, _magnitude_from_thresholds(abs(r), RANK_BISERIAL_THRESHOLDS)


def _get_effect_size_thresholds(effect_size_name: str) -> Dict[str, float]:
    return RANK_BISERIAL_THRESHOLDS if effect_size_name == 'rank-biserial r' else COHENS_D_THRESHOLDS


# ============================================================================
# PAIRED COMPARISON
# ============================================================================

def compare_paired(
    group1_name: str, group1_items: List[ReviewItem],
    group2_name: str, group2_items: List[ReviewItem],
    metric: str,
) -> PairedComparisonResult:
    is_ordinal = (metric == 'significance')

    vals1, pos1, total1 = extract_metric_values(group1_items, metric)
    vals2, pos2, total2 = extract_metric_values(group2_items, metric)

    rate1_items = pos1 / total1 if total1 > 0 else np.nan
    rate2_items = pos2 / total2 if total2 > 0 else np.nan

    if is_ordinal:
        rate1_ci_lo, rate1_ci_hi = bootstrap_ci_mean(vals1)
        rate2_ci_lo, rate2_ci_hi = bootstrap_ci_mean(vals2)
    else:
        rate1_ci_lo, rate1_ci_hi = wilson_ci(pos1, total1)
        rate2_ci_lo, rate2_ci_hi = wilson_ci(pos2, total2)

    paper_rates1 = {k: v for k, v in calculate_paper_rates(group1_items, metric).items() if not np.isnan(v)}
    paper_rates2 = {k: v for k, v in calculate_paper_rates(group2_items, metric).items() if not np.isnan(v)}

    n1_papers, n2_papers = len(paper_rates1), len(paper_rates2)
    paired_papers = sorted(set(paper_rates1.keys()) & set(paper_rates2.keys()))
    n_paired = len(paired_papers)

    effect_name = 'rank-biserial r' if is_ordinal else "Cohen's d (paired)"

    if n_paired < 2:
        return PairedComparisonResult(
            group1=group1_name, group2=group2_name, metric=metric,
            n1_items=total1, n2_items=total2, pos1_items=pos1, pos2_items=pos2,
            rate1_items=rate1_items, rate2_items=rate2_items,
            rate1_ci_lower=rate1_ci_lo, rate1_ci_upper=rate1_ci_hi,
            rate2_ci_lower=rate2_ci_lo, rate2_ci_upper=rate2_ci_hi,
            n1_papers=n1_papers, n2_papers=n2_papers, n_paired_papers=n_paired,
            mean_diff=np.nan, std_diff=np.nan, se_diff=np.nan,
            ci_diff_lower=np.nan, ci_diff_upper=np.nan,
            test_statistic=np.nan, p_value=1.0,
            significant=False,
            effect_size=np.nan, effect_size_name=effect_name,
            effect_magnitude='undetermined',
            direction='undetermined', classification='?',
        )

    paired_vals1 = np.array([paper_rates1[p] for p in paired_papers])
    paired_vals2 = np.array([paper_rates2[p] for p in paired_papers])
    differences = paired_vals1 - paired_vals2

    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))
    se_diff = std_diff / np.sqrt(n_paired)

    t_crit = stats.t.ppf(0.975, df=n_paired - 1)
    ci_diff_lo = mean_diff - t_crit * se_diff
    ci_diff_hi = mean_diff + t_crit * se_diff

    if is_ordinal:
        nonzero = differences[differences != 0]
        if len(nonzero) < 1:
            test_stat, p_value = 0.0, 1.0
        else:
            try:
                test_stat, p_value = wilcoxon(paired_vals1, paired_vals2, alternative='two-sided')
            except ValueError:
                test_stat, p_value = 0.0, 1.0
        d, magnitude = rank_biserial_correlation(differences)
    else:
        test_stat, p_value = ttest_1samp(differences, 0)
        d, magnitude = cohens_d_paired(differences)

    significant = p_value < ALPHA
    if significant:
        direction = 'group1_better' if mean_diff > 0 else 'group2_better'
        classification = ('++' if magnitude in ('medium', 'large') else '+') if mean_diff > 0 \
            else ('--' if magnitude in ('medium', 'large') else '-')
    else:
        direction = 'undetermined'
        classification = '?'

    return PairedComparisonResult(
        group1=group1_name, group2=group2_name, metric=metric,
        n1_items=total1, n2_items=total2, pos1_items=pos1, pos2_items=pos2,
        rate1_items=rate1_items, rate2_items=rate2_items,
        rate1_ci_lower=rate1_ci_lo, rate1_ci_upper=rate1_ci_hi,
        rate2_ci_lower=rate2_ci_lo, rate2_ci_upper=rate2_ci_hi,
        n1_papers=n1_papers, n2_papers=n2_papers, n_paired_papers=n_paired,
        mean_diff=mean_diff, std_diff=std_diff, se_diff=se_diff,
        ci_diff_lower=ci_diff_lo, ci_diff_upper=ci_diff_hi,
        test_statistic=float(test_stat), p_value=float(p_value),
        significant=significant,
        effect_size=float(d), effect_size_name=effect_name,
        effect_magnitude=magnitude,
        direction=direction, classification=classification,
    )


def _update_classification(r: PairedComparisonResult):
    """Classification uses raw p-value + effect size gate."""
    thresholds = _get_effect_size_thresholds(r.effect_size_name)
    if r.significant and not np.isnan(r.effect_size) and abs(r.effect_size) >= thresholds['negligible']:
        if r.mean_diff > 0:
            r.direction = 'group1_better'
            r.classification = '++' if r.effect_magnitude in ('medium', 'large') else '+'
        else:
            r.direction = 'group2_better'
            r.classification = '--' if r.effect_magnitude in ('medium', 'large') else '-'
    else:
        r.direction = 'undetermined' if not r.significant else 'negligible_effect'
        r.classification = '?' if not r.significant else '≈'


def run_all_comparisons(groups: Dict[str, List[ReviewItem]], metric: str) -> List[PairedComparisonResult]:
    group_names = ['Best Human', 'Worst Human', 'GPT', 'Claude', 'Gemini']
    results = []
    for i, g1 in enumerate(group_names):
        for g2 in group_names[i+1:]:
            if g1 not in groups or g2 not in groups:
                continue
            if not groups[g1] or not groups[g2]:
                continue
            results.append(compare_paired(g1, groups[g1], g2, groups[g2], metric))
    # Re-classify using raw p-values + effect size gate
    for r in results:
        _update_classification(r)
    return results


# ============================================================================
# REPORTING
# ============================================================================

def _n_papers(items: List[ReviewItem]) -> int:
    return len({i.paper_id for i in items})


def generate_descriptive_stats_tables(groups: Dict[str, List[ReviewItem]]) -> str:
    lines = []
    group_order = ['Best Human', 'Worst Human', 'GPT', 'Claude', 'Gemini']
    lines.append("")
    lines.append("=" * 110)
    lines.append("DESCRIPTIVE STATISTICS (Item-Level) with 95% Confidence Intervals")
    lines.append("=" * 110)
    lines.append("")

    # ---- Table 1: Correctness ----
    lines.append("TABLE 1: CORRECTNESS RATE (Wilson 95% CI)")
    lines.append("-" * 105)
    lines.append(
        f"{'Group':<15} {'Rate':>9} {'95% CI':>20} {'Correct':>9} {'Incorrect':>11} "
        f"{'Total Items':>13} {'(# papers)':>12}"
    )
    lines.append("-" * 105)
    for g in group_order:
        if g not in groups:
            continue
        items_for_metric = get_items_for_correctness(groups[g])
        _, pos, total = extract_metric_values(groups[g], 'correctness')
        neg = total - pos
        rate = pos / total if total > 0 else 0
        lo, hi = wilson_ci(pos, total)
        ci = f"[{lo:.1%}, {hi:.1%}]" if not np.isnan(lo) else "N/A"
        npapers = _n_papers(items_for_metric)
        lines.append(
            f"{g:<15} {rate:>9.1%} {ci:>20} {pos:>9} {neg:>11} "
            f"{total:>13} {'(' + str(npapers) + ')':>12}"
        )
    lines.append("")

    # ---- Table 2: Significance (3-class) ----
    lines.append("TABLE 2: SIGNIFICANCE DISTRIBUTION (Bootstrap 95% CI for Mean, 0-2 scale)")
    lines.append("-" * 115)
    lines.append(
        f"{'Group':<15} {'Mean':>7} {'95% CI':>16} {'Not Sig.':>10} {'Marg.':>8} "
        f"{'Sig.':>8} {'Total Items':>13} {'(# papers)':>12}"
    )
    lines.append("-" * 115)
    for g in group_order:
        if g not in groups:
            continue
        items_for_metric = get_items_for_significance(groups[g])
        total = len(items_for_metric)
        npapers = _n_papers(items_for_metric)
        if total == 0:
            lines.append(
                f"{g:<15} {'N/A':>7} {'N/A':>16} {0:>10} {0:>8} {0:>8} "
                f"{0:>13} {'(0)':>12}"
            )
            continue
        not_sig = sum(1 for i in items_for_metric if i.significance_numeric == 0)
        marg = sum(1 for i in items_for_metric if i.significance_numeric == 1)
        sig = sum(1 for i in items_for_metric if i.significance_numeric == 2)
        vals = [i.significance_numeric for i in items_for_metric]
        mean_sig = np.mean(vals)
        lo, hi = bootstrap_ci_mean(vals)
        ci = f"[{lo:.2f}, {hi:.2f}]" if not np.isnan(lo) else "N/A"
        lines.append(
            f"{g:<15} {mean_sig:>7.2f} {ci:>16} {not_sig:>10} {marg:>8} {sig:>8} "
            f"{total:>13} {'(' + str(npapers) + ')':>12}"
        )
    lines.append("")

    # ---- Table 3: Evidence ----
    lines.append("TABLE 3: SUFFICIENCY OF EVIDENCE RATE (Wilson 95% CI)")
    lines.append("-" * 110)
    lines.append(
        f"{'Group':<15} {'Rate':>9} {'95% CI':>20} {'Sufficient':>12} {'Not Sufficient':>16} "
        f"{'Total Items':>13} {'(# papers)':>12}"
    )
    lines.append("-" * 110)
    for g in group_order:
        if g not in groups:
            continue
        items_for_metric = get_items_for_evidence(groups[g])
        _, pos, total = extract_metric_values(groups[g], 'evidence')
        neg = total - pos
        rate = pos / total if total > 0 else 0
        lo, hi = wilson_ci(pos, total)
        ci = f"[{lo:.1%}, {hi:.1%}]" if not np.isnan(lo) else "N/A"
        npapers = _n_papers(items_for_metric)
        lines.append(
            f"{g:<15} {rate:>9.1%} {ci:>20} {pos:>12} {neg:>16} "
            f"{total:>13} {'(' + str(npapers) + ')':>12}"
        )
    lines.append("")
    return "\n".join(lines)


def generate_pairwise_table(results: List['PairedComparisonResult'], metric_name: str) -> str:
    """Pretty-print pairwise comparisons as a single table (replaces the 5×5
    comparison matrix). Columns: comparison, diff with CI, p-value, effect
    size + magnitude, paired-paper count."""
    is_ordinal = (metric_name.lower() == 'significance')
    effect_label = 'r (rank-biserial)' if is_ordinal else "d (Cohen's d)"
    lines = []
    lines.append(f"\nPAIRWISE COMPARISONS — {metric_name.upper()}")
    lines.append("-" * 125)
    lines.append(
        f"{'Comparison':<28} {'Diff':>9} {'95% CI (diff)':>22} {'p-value':>12} "
        f"{effect_label:>20} {'magnitude':>12} {'N(paired)':>10}"
    )
    lines.append("-" * 125)
    for r in sorted(results, key=lambda x: (np.inf if np.isnan(x.p_value) else x.p_value)):
        cmp = f"{r.group1} vs {r.group2}"
        if np.isnan(r.mean_diff):
            diff_s, ci_s = "N/A", "N/A"
        elif is_ordinal:
            diff_s = f"{r.mean_diff:+.2f}"
            ci_s = f"[{r.ci_diff_lower:+.2f}, {r.ci_diff_upper:+.2f}]"
        else:
            diff_s = f"{r.mean_diff:+.1%}"
            ci_s = f"[{r.ci_diff_lower:+.1%}, {r.ci_diff_upper:+.1%}]"
        p_s = f"{r.p_value:.4f}" if not np.isnan(r.p_value) else "N/A"
        if r.significant:
            p_s += " *"
        es = f"{r.effect_size:+.3f}" if not np.isnan(r.effect_size) else "N/A"
        lines.append(
            f"{cmp:<28} {diff_s:>9} {ci_s:>22} {p_s:>12} "
            f"{es:>20} {r.effect_magnitude:>12} {r.n_paired_papers:>10}"
        )
    lines.append("")
    lines.append(f"  * = p < {ALPHA}")
    lines.append("")
    return "\n".join(lines)


def generate_report(
    groups: Dict[str, List[ReviewItem]],
    correctness_results: List[PairedComparisonResult],
    significance_results: List[PairedComparisonResult],
    evidence_results: List[PairedComparisonResult],
) -> str:
    lines = []
    lines.append("=" * 110)
    lines.append("PEERREVIEW BENCH — STATISTICAL ANALYSIS REPORT (PAIRED)")
    lines.append("=" * 110)
    lines.append("")
    lines.append("METHODOLOGY")
    lines.append("-" * 110)
    lines.append("")
    lines.append("  Data source: the `expert_annotation` config of the `prometheus-eval/peerreview-bench`")
    lines.append("  HuggingFace dataset. Both primary and secondary annotators' rows are included as")
    lines.append("  independent data points (merged semantics).")
    lines.append("")
    lines.append("  Reviewer groups:")
    lines.append("    Best Human, Worst Human, GPT, Claude, Gemini (5 groups total).")
    lines.append("    'Best/Worst Human' comes from the reviewer_rankings file, per paper.")
    lines.append("")
    lines.append("  Per-metric test, effect size, and confidence interval:")
    lines.append("")
    lines.append("    Correctness (binary: Correct / Not Correct)")
    lines.append("      - Item-level rate CI:   Wilson score interval")
    lines.append("      - Paired paper-level:   paired t-test on per-paper rate differences")
    lines.append("      - Effect size:          Cohen's d (paired) = mean(diff) / SD(diff)")
    lines.append("      - Diff CI:              t-interval on the paired paper-level differences")
    lines.append("")
    lines.append("    Significance (ordinal 3-class: Not Sig / Marginally Sig / Significant, encoded 0/1/2)")
    lines.append("      - Item-level mean CI:   non-parametric bootstrap (10,000 resamples)")
    lines.append("      - Paired paper-level:   Wilcoxon signed-rank on per-paper mean differences")
    lines.append("      - Effect size:          rank-biserial r (Kerby 2014)")
    lines.append("      - Diff CI:              t-interval on the paired paper-level differences")
    lines.append("")
    lines.append("    Evidence (binary: Sufficient / Requires More)")
    lines.append("      - Item-level rate CI:   Wilson score interval")
    lines.append("      - Paired paper-level:   paired t-test on per-paper rate differences")
    lines.append("      - Effect size:          Cohen's d (paired) = mean(diff) / SD(diff)")
    lines.append("      - Diff CI:              t-interval on the paired paper-level differences")
    lines.append("")
    lines.append(f"  Significance level: α = {ALPHA}. Raw p-values are reported.")
    lines.append("")

    lines.append("SAMPLE SIZES (per reviewer group)")
    lines.append("-" * 110)
    for name, items in groups.items():
        n_papers = len(set(i.paper_id for i in items))
        lines.append(f"  {name:15} n_items={len(items):4}  n_papers={n_papers:3}")
    lines.append("")

    lines.append(generate_descriptive_stats_tables(groups))

    lines.append("")
    lines.append("Note: per-comparison pairwise paired tests (p-values, effect sizes, 95% CIs")
    lines.append("on the diff) are identical between this report and the per-paper report in")
    lines.append("`analysis_output_per_paper/analysis_report_per_paper.txt`; they are reported")
    lines.append("only in that file to avoid duplication. The pairwise-comparison CSVs")
    lines.append("(`{correctness,significance,evidence}_comparisons.csv`) are still written")
    lines.append("alongside this report for programmatic access.")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def run_analysis(items: List[ReviewItem], rankings: Dict[int, Dict[str, str]],
                 output_dir: Optional[str] = None) -> Dict[str, Any]:
    print("\n" + "=" * 60)
    print("PEERREVIEW BENCH — STATISTICAL ANALYSIS (PAIRED)")
    print("=" * 60)

    # --- Dataset overview (no rule jargon) ---
    print("\n1. Computing dataset overview...")
    overview = summarize_dataset(items)
    overview_report = format_dataset_overview(overview)
    print(overview_report)

    kept_items = filter_kept_items(items)

    print("\n2. Creating comparison groups...")
    groups = create_comparison_groups(kept_items, rankings)
    for name, g_items in groups.items():
        n_papers = len(set(i.paper_id for i in g_items))
        print(f"   {name:15} n_items={len(g_items):4}, n_papers={n_papers:3}")

    # --- Fully-good stats per group ---
    print("\n3. Computing fully-good stats per group...")
    fully_good = compute_fully_good_stats(groups)
    fully_good_report = format_fully_good_report(fully_good)
    print(fully_good_report)

    print("\n4. Running PAIRED pairwise comparisons...")
    correctness_results = run_all_comparisons(groups, 'correctness')
    significance_results = run_all_comparisons(groups, 'significance')
    evidence_results = run_all_comparisons(groups, 'evidence')

    print("\n5. Generating report...")
    report = (overview_report + "\n" + fully_good_report + "\n" +
              generate_report(groups, correctness_results, significance_results, evidence_results))
    print(report)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / 'analysis_report.txt', 'w') as f:
            f.write(report)
        import json as _json
        with open(out / 'dataset_overview.json', 'w') as f:
            _json.dump(overview, f, indent=2, default=str)
        with open(out / 'fully_good_stats.json', 'w') as f:
            _json.dump(fully_good, f, indent=2)
        for name, results in [('correctness', correctness_results),
                              ('significance', significance_results),
                              ('evidence', evidence_results)]:
            rows = [{
                'group1': r.group1, 'group2': r.group2, 'metric': r.metric,
                'n1_items': r.n1_items, 'n2_items': r.n2_items,
                'rate1_items': r.rate1_items, 'rate2_items': r.rate2_items,
                'rate1_ci_lower': r.rate1_ci_lower, 'rate1_ci_upper': r.rate1_ci_upper,
                'rate2_ci_lower': r.rate2_ci_lower, 'rate2_ci_upper': r.rate2_ci_upper,
                'n_paired_papers': r.n_paired_papers,
                'mean_diff': r.mean_diff, 'std_diff': r.std_diff,
                'ci_diff_lower': r.ci_diff_lower, 'ci_diff_upper': r.ci_diff_upper,
                'p_value': r.p_value,
                'effect_size': r.effect_size, 'effect_magnitude': r.effect_magnitude,
            } for r in results]
            pd.DataFrame(rows).to_csv(out / f'{name}_comparisons.csv', index=False)
        print(f"\n6. Results saved to: {output_dir}")

    return {
        'groups': {k: len(v) for k, v in groups.items()},
        'correctness_results': correctness_results,
        'significance_results': significance_results,
        'evidence_results': evidence_results,
        'report': report,
    }


def main():
    parser = argparse.ArgumentParser(description='PeerReview Bench Analysis (PAIRED)')
    parser.add_argument(
        '--output-dir', type=str,
        default=str(Path(__file__).resolve().parent.parent
                    / 'outputs' / 'analysis' / 'analysis_output'),
    )
    parser.add_argument('--annotator-source', type=str, default='both',
                        choices=['primary', 'secondary', 'both'],
                        help="Which set of meta-reviewer annotations to use. "
                             "Default 'both' merges primary and secondary as "
                             "independent data points.")
    args = parser.parse_args()

    print(f"Loading annotations (annotator_source={args.annotator_source}) from HuggingFace dataset...")
    items, rankings = load_annotations(annotator_source=args.annotator_source)
    print(f"Loaded {len(items)} items, {len(rankings)} paper rankings")

    run_analysis(items, rankings, args.output_dir)
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
